import os
import torch
from torch import nn
import torch.distributed as dist
import math

# FSDP (fully sharded data paralellism)

class GatherShardedWeight(torch.autograd.Function): 
  # pytorch autograd doesnt know how to backprop through dist.allgather, so we write a wrapper class
  @staticmethod
  def forward(ctx, shard, shape, numel):
      return gather_params(shard, shape, numel) #reconstructs the weights

  @staticmethod
  def backward(ctx, grad_output):
      grad_shard = reduce_scatter_gradients(grad_output) # gets gradient, then reduce scatters so all ranks have their own gradients

      return grad_shard, None, None

class ShardedLinear(nn.Module):
  def __init__(self, linearlayer: nn.Linear):
    super().__init__()

    if linearlayer.bias is not None: raise ValueError ('only bias = false supported rn')

    shard, shape, numel = parameter_sharding(linearlayer.weight)

    self.weightshard = nn.Parameter(shard)
    self.shape = shape
    self.numel = numel

    self.in_features, self.out_features = linearlayer.in_features, linearlayer.out_features
  
  def forward(self, x):
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel)

    # x * W + b, except no b
    output = nn.functional.linear(input=x, weight=weights, bias = None)

    return output

def parameter_sharding(parameter_tensor):
  total_nodes = dist.get_world_size()

  # flatten it and split into chunks

  shape, numel = parameter_tensor.shape, parameter_tensor.numel()

  params = parameter_tensor.reshape(-1)

  shardsize = math.ceil(params.shape[0] / total_nodes)

  if params.numel() != shardsize * total_nodes:
    # pad with sufficient zeros

    amtmissing = shardsize * total_nodes - params.numel()
    params = nn.functional.pad(params, (0, amtmissing))

  params = params.split(shardsize)

  rank = dist.get_rank()

  myshard = params[rank].detach().clone()

  return myshard, shape, numel

def gather_params(myshard, shape, numel):
  # reconstruct the original tensor given shards

  total_nodes = dist.get_world_size()

  # thus we know the size of our tensor is going to be worldsize * shardsize

  allshards = [torch.empty_like(myshard) for i in range(total_nodes)]

  dist.all_gather(allshards, myshard)

  allshards = torch.cat(allshards, dim = 0)
  allshards = allshards[:numel]
  allshards = allshards.reshape(shape)

  return allshards

def reduce_scatter_gradients(gradients):
  total_nodes = dist.get_world_size()

  gradients = gradients.reshape(-1)

  shardsize = math.ceil(gradients.shape[0] / total_nodes)

  if gradients.numel() != shardsize * total_nodes:
    # pad with sufficient zeros

    amtmissing = shardsize * total_nodes - gradients.numel()
    gradients = nn.functional.pad(gradients, (0, amtmissing))

  output = torch.empty((shardsize,), dtype=gradients.dtype, device=gradients.device)

  gradients = gradients.contiguous() 

  dist.reduce_scatter_tensor(output=output, input=gradients, op=dist.ReduceOp.SUM)

  output.div_(total_nodes)

  return output

def shard_model(model):
  model.outputprojection = ShardedLinear(model.outputprojection)

  return model
