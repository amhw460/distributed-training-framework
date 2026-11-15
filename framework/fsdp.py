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

class ShardedEmbedding(nn.Module):
  def __init__(self, embed: nn.Embedding):

    shard, shape, numel = parameter_sharding(embed.weight)

    self.weightshard = nn.Parameter(shard)
    self.shape = shape
    self.numel = numel

  def forward(self, token_ids):
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel)
        
    return nn.functional.embedding(token_ids, weights)

class ShardedRMSNorm(nn.Module):
  def __init__(self, norm: nn.RMSNorm):
    super().__init__()

    shard, shape, numel = parameter_sharding(norm.weight)

    self.weightshard = nn.Parameter(shard)
    self.shape = shape
    self.numel = numel

    self.normalized_shape = norm.normalized_shape
    self.eps = norm.eps

    def forward(self, x):
      weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel)

      return nn.functional.rms_norm(x, self.normalized_shape, weight=weights, eps=self.eps)

class ShardedExperts(nn.Module):
  def __init__(self, experts: nn.ModuleList):
    super().__init__()

    self.num_experts = len(experts)

    self.weight_shapes = []
    self.weight_numels = []
    flattened_weights = []

    for expert in experts:
        for linear in (expert.up_proj, expert.gate, expert.down_proj):
          weight = linear.weight

          self.weight_shapes.append(weight.shape)
          self.weight_numels.append(weight.numel())

          flattened_weights.append(weight.detach().reshape(-1))

        # Combine every expert's weights into one tensor
    combined_weights = torch.cat(flattened_weights)

        # Each rank permanently owns only its shard
    shard, shape, numel = parameter_sharding(combined_weights)

    self.weightshard = nn.Parameter(shard)
    self.shape = shape
    self.numel = numel

  def forward(self, tokenstates, topkindices, topkweights):
    # Gather all expert weights once, across all ranks
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel)

    # Recover the individual weight matrices
    matrices = torch.split(weights, self.weight_numels)
    matrices = [matrix.reshape(shape) for matrix, shape in zip(matrices, self.weight_shapes)]

    output = torch.zeros_like(tokenstates)

    for expert_idx in range(self.num_experts):
      tokenindices, whichoftopk = torch.where(topkindices == expert_idx)

      if tokenindices.numel() == 0: continue

        # Three matrices per expert, in constructor order
      up_weight = matrices[3 * expert_idx]
      gate_weight = matrices[3 * expert_idx + 1]
      down_weight = matrices[3 * expert_idx + 2]

      tokens = tokenstates[tokenindices]

        # SwiGLU expert computation
      up = nn.functional.linear(tokens, up_weight)
      gate = nn.functional.silu(nn.functional.linear(tokens, gate_weight))

      expert_output = nn.functional.linear(up * gate, down_weight)

        # Apply router weights
      router_weights = topkweights[tokenindices, whichoftopk].unsqueeze(-1)

      output = output.index_add(0, tokenindices, router_weights * expert_output)

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
  model.embeddings = ShardedEmbedding(model.embeddings)
  model.finalRMSnorm = ShardedRMSNorm(model.finalRMSnorm)

  for layer in model.transformerlayers:

    layer.attention.Wq = ShardedLinear(layer.attention.Wq)
    layer.attention.Wk = ShardedLinear(layer.attention.Wk)
    layer.attention.Wv = ShardedLinear(layer.attention.Wv)
    layer.attention.output = ShardedLinear(layer.attention.output)

    layer.firstnorm = ShardedRMSNorm(layer.firstnorm)
    layer.secondnorm = ShardedRMSNorm(layer.secondnorm)

    layer.attention.rmsNormQ = ShardedRMSNorm(layer.attention.rmsNormQ)
    layer.attention.rmsNormK = ShardedRMSNorm(layer.attention.rmsNormK)

    layer.moe.router.router = ShardedLinear(layer.moe.router.router)
    layer.moe.experts = ShardedExperts(layer.moe.experts)

  return model
