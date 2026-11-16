import torch
from torch import nn
import torch.distributed as dist
import os

# tp splits weight matrices across ranks and syncs the partial results

class CopyToTensorParallelRegion(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, group):
      ctx.group = group
      return x

  @staticmethod
  def backward(ctx, grad_output):
      grad_input = grad_output.contiguous()
      dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
      return grad_input, None


class ColumnwiseLinearParallelism(nn.Module):

  # each rank owns a slice of the output features

  def __init__(self, linear: nn.Linear, group=None):
    super().__init__()

    total_nodes = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    self.in_features, self.out_features = linear.in_features, linear.out_features

    if linear.bias is not None: raise ValueError("e")

    self.rank = rank
    self.nodes = total_nodes
    self.group = group

    chunks = torch.tensor_split(linear.weight.detach(), self.nodes, dim = 0)

    this_weight = chunks[self.rank].clone()

    self.weight = nn.Parameter(this_weight)
    self.local_out_features = this_weight.shape[0]

  def forward(self, x):
    x = CopyToTensorParallelRegion.apply(x, self.group)
    return nn.functional.linear(x, self.weight, bias=None)

# rowwise tp sums partial outputs after the matmul
class ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        output = x.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class RowwiseLinearParallelism(nn.Module):

  # each rank owns a slice of the input features

  def __init__(self, linear: nn.Linear, group=None):
    super().__init__()

    total_nodes = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    self.in_features, self.out_features = linear.in_features, linear.out_features

    if linear.bias is not None: raise ValueError("e")

    self.rank = rank
    self.nodes = total_nodes
    self.group = group

    chunks = torch.tensor_split(linear.weight.detach(), self.nodes, dim = 1)

    this_weight = chunks[self.rank].clone()

    self.weight = nn.Parameter(this_weight)
    self.local_in_features = this_weight.shape[1]

  def forward(self, x):
    output = nn.functional.linear(x, self.weight, bias=None)
    return ReduceFromTensorParallelRegion.apply(output, self.group)

# split q/k/v by output heads and the attention output projection by input heads

def parallelize_attn(attention, group=None):
  world_size = dist.get_world_size(group=group)

  if (attention.model_args.numheads % world_size != 0 or attention.model_args.numkvheads % world_size != 0):
      raise ValueError("q and kv head counts must be divisible by world size")

  attention.Wq = ColumnwiseLinearParallelism(attention.Wq, group=group)
  attention.Wk = ColumnwiseLinearParallelism(attention.Wk, group=group)
  attention.Wv = ColumnwiseLinearParallelism(attention.Wv, group=group)

  attention.output = RowwiseLinearParallelism(attention.output, group=group)

  attention.local_numheads = attention.Wq.local_out_features // attention.model_args.head_dim
  attention.local_numkvheads = attention.Wk.local_out_features // attention.model_args.head_dim

  return attention

def parallelize_moe(moe, group=None):
  for expert in moe.experts:
        expert.up_proj = ColumnwiseLinearParallelism(expert.up_proj, group=group)
        expert.gate = ColumnwiseLinearParallelism(expert.gate, group=group)
        expert.down_proj = RowwiseLinearParallelism(expert.down_proj, group=group)

  return moe

def paralellize_model(model, group=None):
  for layer in model.transformerlayers:
        parallelize_attn(layer.attention, group=group)
        parallelize_moe(layer.moe, group=group)

  return model


def synchronize_tp_head_norm_gradients(model, group=None):
  """Sum gradients for q/k norms shared across tp ranks."""
  for layer in model.transformerlayers:
    attention = layer.attention
    for norm in (attention.rmsNormQ, attention.rmsNormK):
      parameter = norm.weightshard if hasattr(norm, "weightshard") else norm.weight
      if parameter.grad is None:
        raise RuntimeError("Expected Q/K norm gradient during TP training")
      dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
