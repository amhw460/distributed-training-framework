import torch
from torch import nn
import torch.distributed as dist
import os

# tensor parallelism splits a layers output, this implementation is doing it togehter
# so like if u have a -> b in terms of layers u split a in terms of the output (column) and b in terms of the input (row)
# as to efficiently paralellize work


# wrapper class exists to add the gradient contributions from ranks and make sure that u have
# correct gradient

class CopyToTensorParallelRegion(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
      return x

  @staticmethod
  def backward(ctx, grad_output):
      grad_input = grad_output.contiguous()
      dist.all_reduce(grad_input, op=dist.ReduceOp.SUM)
      return grad_input


class ColumnwiseLinearParallelism(nn.Module):

  # tensor parallelism except by column (splitting across output features), just wrap a linear layer 
  # and get the chunks that you are accounting for

  def __init__(self, linear: nn.Linear):
    super().__init__()

    total_nodes = dist.get_world_size()
    rank = dist.get_rank()

    self.in_features, self.out_features = linear.in_features, linear.out_features

    if linear.bias is not None: raise ValueError("e")

    self.rank = dist.get_rank()
    self.nodes = dist.get_world_size()

    chunks = torch.tensor_split(linear.weight.detach(), self.nodes, dim = 0)

    this_weight = chunks[self.rank].clone()

    self.weight = nn.Parameter(this_weight)
    self.local_out_features = this_weight.shape[0]

  def forward(self, x):
    x = CopyToTensorParallelRegion.apply(x)
    return nn.functional.linear(x, self.weight, bias=None)

# ditto but for row wise split
class ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = x.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class RowwiseLinearParallelism(nn.Module):

  # tensor parallelism except by column (splitting across output features), just wrap a linear layer 
  # and get the chunks that you are accounting for

  def __init__(self, linear: nn.Linear):
    super().__init__()

    total_nodes = dist.get_world_size()
    rank = dist.get_rank()

    self.in_features, self.out_features = linear.in_features, linear.out_features

    if linear.bias is not None: raise ValueError("e")

    self.rank = dist.get_rank()
    self.nodes = dist.get_world_size()

    chunks = torch.tensor_split(linear.weight.detach(), self.nodes, dim = 1)

    this_weight = chunks[self.rank].clone()

    self.weight = nn.Parameter(this_weight)
    self.local_in_features = this_weight.shape[1]

  def forward(self, x):
    output = nn.functional.linear(x, self.weight, bias=None)
    return ReduceFromTensorParallelRegion.apply(output)

#qkv is largely independent computation, output projection maps attention back to wahtever dim,
# so if ur attention is split amongst outputs logically ur gonna split ur output amongst inputs

def parallelize_attn(attention):
  world_size = dist.get_world_size()

  if (attention.model_args.numheads % world_size != 0 or attention.model_args.numkvheads % world_size != 0):
      raise ValueError("q and kv head counts must be divisible by world size")

  attention.Wq = ColumnwiseLinearParallelism(attention.Wq)
  attention.Wk = ColumnwiseLinearParallelism(attention.Wk)
  attention.Wv = ColumnwiseLinearParallelism(attention.Wv)

  attention.output = RowwiseLinearParallelism(attention.output)

  attention.local_numheads = attention.Wq.local_out_features // attention.model_args.head_dim
  attention.local_numkvheads = attention.Wk.local_out_features // attention.model_args.head_dim

  return attention
  
def parallelize_moe(moe):
  for expert in moe.experts:
    expert.up_proj = ColumnwiseLinearParallelism(up_proj)
    expert.gate = ColumnwiseLinearParallelism(gate)
    expert.down_proj = RowwiseLinearParallelism(expert.down_proj)

  return moe

def paralellize_model(model):
  for layer in model.layers:
    paralellize_attn(layer.attention)
    parallelize_moe(layer.moe)

  return 
  


