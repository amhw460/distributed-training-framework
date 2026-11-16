import torch
from torch import nn
import torch.distributed as dist


class AllToAll(torch.autograd.Function):
  # all_to_all needs its own backward to return activation gradients
  @staticmethod
  def forward(ctx, tokens, sendcounts, recvcounts, group):
    ctx.sendcounts = sendcounts
    ctx.recvcounts = recvcounts
    ctx.group = group

    received = torch.empty((sum(recvcounts), *tokens.shape[1:]), dtype=tokens.dtype, device=tokens.device)
    dist.all_to_all_single(received, tokens.contiguous(),
                          output_split_sizes=recvcounts, input_split_sizes=sendcounts, group=group)
    return received

  @staticmethod
  def backward(ctx, gradients):
    returned = torch.empty((sum(ctx.sendcounts), *gradients.shape[1:]),
                           dtype=gradients.dtype, device=gradients.device)
    dist.all_to_all_single(returned, gradients.contiguous(),
                          output_split_sizes=ctx.sendcounts, input_split_sizes=ctx.recvcounts, group=ctx.group)
    return returned, None, None, None


class ExpertParallel(nn.Module):
  def __init__(self, moe, rank, total_nodes, group=None):
    super().__init__()

    self.rank = rank
    self.total_nodes = total_nodes
    self.group = group
    self.model_args = moe.model_args

    if not 0 <= rank < total_nodes: raise ValueError("invalid rank")
    if self.model_args.experts % total_nodes:
      raise ValueError("number of experts must be divisible by total_nodes")

    self.experts_per_rank = self.model_args.experts // total_nodes
    self.start = rank * self.experts_per_rank

    # router still chooses from all global experts; weights live only on their owner
    self.router = moe.router
    self.experts = nn.ModuleList(list(moe.experts[self.start:self.start + self.experts_per_rank]))

  def forward(self, tokens):
    shape = tokens.shape
    tokenstates = tokens.reshape(-1, shape[-1])
    topkweights, topkindices, _ = self.router(tokenstates)

    # Every selected expert gets a copy of its token, grouped by owning rank.
    expertids = topkindices.reshape(-1)
    destinations = expertids // self.experts_per_rank
    order = torch.argsort(destinations, stable=True)

    sendcounts = torch.bincount(destinations, minlength=self.total_nodes).tolist()
    counts = torch.tensor(sendcounts, dtype=torch.long, device=tokens.device)
    receivedcounts = torch.empty_like(counts)
    dist.all_to_all_single(receivedcounts, counts, group=self.group)
    recvcounts = receivedcounts.tolist()

    sentids = expertids[order].contiguous()
    receivedids = torch.empty(sum(recvcounts), dtype=torch.long, device=tokens.device)
    dist.all_to_all_single(receivedids, sentids,
                          output_split_sizes=recvcounts, input_split_sizes=sendcounts, group=self.group)

    senttokens = tokenstates.repeat_interleave(self.model_args.k, dim=0)[order]
    received = AllToAll.apply(senttokens, sendcounts, recvcounts, self.group)

    # Run only experts assigned to this rank.
    # Multiplying by zero keeps the backward communication alive even if
    # this rank receives no tokens for any local expert.
    localoutput = received * 0
    for i, expert in enumerate(self.experts):
      indices = torch.where(receivedids == self.start + i)[0]
      if indices.numel() == 0 and not getattr(self, 'fsdp_mode', False): continue
      localoutput = localoutput.index_add(0, indices, expert(received[indices]))

    # Send expert outputs back, restore top-k order, then weighted-sum per token.
    returned = AllToAll.apply(localoutput, recvcounts, sendcounts, self.group)
    returned = returned[torch.argsort(order)]
    tokenindices = torch.arange(tokenstates.shape[0], device=tokens.device).repeat_interleave(self.model_args.k)
    output = torch.zeros_like(tokenstates).index_add(
      0, tokenindices, returned * topkweights.reshape(-1, 1).to(returned.dtype)
    )
    return output.reshape(shape)


def parallelize_ep(model, group=None):
  rank, total_nodes = dist.get_rank(group=group), dist.get_world_size(group=group)
  for layer in model.transformerlayers:
    layer.moe = ExpertParallel(layer.moe, rank, total_nodes, group=group)
  return model


def synchronize_ep_gradients(model, group=None):
  # Experts already received gradient contributions from all token-owning ranks.
  # Everything else is replicated and must be all-reduced as in DDP.
  total_nodes = dist.get_world_size(group=group)
  expertparams = {id(p) for module in model.modules() if isinstance(module, ExpertParallel)
                  for expert in module.experts for p in expert.parameters()}

  for param in model.parameters():
    if id(param) in expertparams:
      if param.grad is not None: param.grad.div_(total_nodes)
      continue

    used = torch.tensor(int(param.grad is not None), device=param.device)
    dist.all_reduce(used, op=dist.ReduceOp.SUM, group=group)
    if used.item() == 0: continue

    gradient = param.grad if param.grad is not None else torch.zeros_like(param)
    dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=group)
    gradient.div_(total_nodes)
    if param.grad is None: param.grad = gradient


def clip_ep_grad_norm_(model, max_norm, group=None):
  # Count replicated weights once, and expert weights only on their owner.
  expertparams = {id(p) for module in model.modules() if isinstance(module, ExpertParallel)
                  for expert in module.experts for p in expert.parameters()}
  parameters = list(model.parameters())
  normsq = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
  for param in parameters:
    if param.grad is None: continue
    contribution = param.grad.detach().double().square().sum()
    normsq += contribution if id(param) in expertparams else contribution / dist.get_world_size(group=group)

  dist.all_reduce(normsq, op=dist.ReduceOp.SUM, group=group)
  norm = normsq.sqrt().item()
  coefficient = min(1.0, max_norm / (norm + 1e-6))
  with torch.no_grad():
    for param in parameters:
      if param.grad is not None: param.grad.mul_(coefficient)
  return norm
