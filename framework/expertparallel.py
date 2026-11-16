import torch
from torch import nn
import torch.distributed as dist


class AllToAll(torch.autograd.Function):
  # all_to_all needs its own backward to return activation gradients
  @staticmethod
  def forward(ctx, tokens, sendcounts, recvcounts):
    ctx.sendcounts = sendcounts
    ctx.recvcounts = recvcounts

    received = torch.empty((sum(recvcounts), *tokens.shape[1:]), dtype=tokens.dtype, device=tokens.device)
    dist.all_to_all_single(received, tokens.contiguous(),
                          output_split_sizes=recvcounts, input_split_sizes=sendcounts)
    return received

  @staticmethod
  def backward(ctx, gradients):
    returned = torch.empty((sum(ctx.sendcounts), *gradients.shape[1:]),
                           dtype=gradients.dtype, device=gradients.device)
    dist.all_to_all_single(returned, gradients.contiguous(),
                          output_split_sizes=ctx.sendcounts, input_split_sizes=ctx.recvcounts)
    return returned, None, None


class ExpertParallel(nn.Module):
  def __init__(self, moe, rank, total_nodes):
    super().__init__()

    self.rank = rank
    self.total_nodes = total_nodes
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
    dist.all_to_all_single(receivedcounts, counts)
    recvcounts = receivedcounts.tolist()

    sentids = expertids[order].contiguous()
    receivedids = torch.empty(sum(recvcounts), dtype=torch.long, device=tokens.device)
    dist.all_to_all_single(receivedids, sentids,
                          output_split_sizes=recvcounts, input_split_sizes=sendcounts)

    senttokens = tokenstates.repeat_interleave(self.model_args.k, dim=0)[order]
    received = AllToAll.apply(senttokens, sendcounts, recvcounts)

    # Run only experts assigned to this rank.
    # Multiplying by zero keeps the backward communication alive even if
    # this rank receives no tokens for any local expert.
    localoutput = received * 0
    for i, expert in enumerate(self.experts):
      indices = torch.where(receivedids == self.start + i)[0]
      if indices.numel() == 0: continue
      localoutput = localoutput.index_add(0, indices, expert(received[indices]))

    # Send expert outputs back, restore top-k order, then weighted-sum per token.
    returned = AllToAll.apply(localoutput, recvcounts, sendcounts)
    returned = returned[torch.argsort(order)]
    tokenindices = torch.arange(tokenstates.shape[0], device=tokens.device).repeat_interleave(self.model_args.k)
    output = torch.zeros_like(tokenstates).index_add(
      0, tokenindices, returned * topkweights.reshape(-1, 1).to(returned.dtype)
    )
    return output.reshape(shape)


def parallelize_ep(model):
  rank, total_nodes = dist.get_rank(), dist.get_world_size()
  for layer in model.transformerlayers:
    layer.moe = ExpertParallel(layer.moe, rank, total_nodes)
  return model


def synchronize_ep_gradients(model):
  # Experts already received gradient contributions from all token-owning ranks.
  # Everything else is replicated and must be all-reduced as in DDP.
  total_nodes = dist.get_world_size()
  expertparams = {id(p) for module in model.modules() if isinstance(module, ExpertParallel)
                  for expert in module.experts for p in expert.parameters()}

  for param in model.parameters():
    if id(param) in expertparams:
      if param.grad is not None: param.grad.div_(total_nodes)
      continue

    used = torch.tensor(int(param.grad is not None), device=param.device)
    dist.all_reduce(used, op=dist.ReduceOp.SUM)
    if used.item() == 0: continue

    gradient = param.grad if param.grad is not None else torch.zeros_like(param)
    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    gradient.div_(total_nodes)
    if param.grad is None: param.grad = gradient
