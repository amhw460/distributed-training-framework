import torch
import torch.distributed as dist


def rotate(*tensors):
  # Move K/V (and during backward, their gradients) to the next rank.
  rank, total_nodes = dist.get_rank(), dist.get_world_size()
  if total_nodes == 1: return tensors

  next_rank = (rank + 1) % total_nodes
  previous_rank = (rank - 1) % total_nodes
  received = [torch.empty_like(tensor) for tensor in tensors]
  ops = []
  for i, (tensor, buffer) in enumerate(zip(tensors, received)):
    ops.append(dist.P2POp(dist.isend, tensor.contiguous(), next_rank, tag=i))
    ops.append(dist.P2POp(dist.irecv, buffer, previous_rank, tag=i))
  for request in dist.batch_isend_irecv(ops):
    request.wait()
  return tuple(received)


class RingAttention(torch.autograd.Function):
  @staticmethod
  def forward(ctx, q, k, v):
    rank, total_nodes = dist.get_rank(), dist.get_world_size()
    scale = q.shape[-1] ** -0.5
    query = q.float()

    # Online softmax: running max, denominator, and weighted values.
    maximum = torch.full(q.shape[:-1], -float("inf"), device=q.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query)

    currentk, currentv = k.float().contiguous(), v.float().contiguous()
    for step in range(total_nodes):
      owner = (rank - step) % total_nodes
      if owner <= rank:  # later tokens cannot attend to future chunks
        scores = query @ currentk.transpose(-2, -1) * scale
        if owner == rank:
          future = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
          scores = scores.masked_fill(future, -float("inf"))

        newmax = torch.maximum(maximum, scores.amax(dim=-1))
        oldscale = torch.exp(maximum - newmax)
        weights = torch.exp(scores - newmax.unsqueeze(-1))

        denominator = denominator * oldscale + weights.sum(dim=-1)
        numerator = numerator * oldscale.unsqueeze(-1) + weights @ currentv
        maximum = newmax

      if step + 1 < total_nodes:
        currentk, currentv = rotate(currentk, currentv)

    output = numerator / denominator.unsqueeze(-1)
    ctx.save_for_backward(q, k, v, output, maximum + denominator.log())
    return output.to(q.dtype)

  @staticmethod
  def backward(ctx, grad_output):
    q, k, v, output, logsumexp = ctx.saved_tensors
    rank, total_nodes = dist.get_rank(), dist.get_world_size()
    scale = q.shape[-1] ** -0.5

    query = q.float()
    gradient = grad_output.float()
    delta = (gradient * output).sum(dim=-1, keepdim=True)
    dq = torch.zeros_like(query)

    currentk, currentv = k.float().contiguous(), v.float().contiguous()
    dk, dv = torch.zeros_like(currentk), torch.zeros_like(currentv)

    for step in range(total_nodes):
      owner = (rank - step) % total_nodes
      if owner <= rank:
        scores = query @ currentk.transpose(-2, -1) * scale
        if owner == rank:
          future = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
          scores = scores.masked_fill(future, -float("inf"))

        probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))
        dp = gradient @ currentv.transpose(-2, -1)
        ds = probabilities * (dp - delta)

        dq += ds @ currentk * scale
        dk += ds.transpose(-2, -1) @ query * scale
        dv += probabilities.transpose(-2, -1) @ gradient

      # The gradients travel with their K/V block, accumulating contributions
      # from every querying rank. The last rotation returns them to their owner.
      if total_nodes > 1:
        currentk, currentv, dk, dv = rotate(currentk, currentv, dk, dv)

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def ring_attention(q, k, v):
  return RingAttention.apply(q, k, v)


def parallelize_cp(model):
  for layer in model.transformerlayers:
    layer.attention.cp_mode = True
  return model
