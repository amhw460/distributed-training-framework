import os
import torch
from torch import nn
import torch.distributed as dist
import math

# FSDP (fully sharded data paralellism)

class GatherShardedWeight(torch.autograd.Function): 
  # pytorch autograd doesnt know how to backprop through dist.allgather, so we write a wrapper class
  @staticmethod
  def forward(ctx, shard, shape, numel, group):
      ctx.group = group
      return gather_params(shard, shape, numel, group=group) #reconstructs the weights

  @staticmethod
  def backward(ctx, grad_output):
      grad_shard = reduce_scatter_gradients(grad_output, group=ctx.group) # gets gradient, then reduce scatters so all ranks have their own gradients

      return grad_shard, None, None, None

class ShardedLinear(nn.Module):
  def __init__(self, linearlayer: nn.Linear, group=None):
    super().__init__()

    if linearlayer.bias is not None: raise ValueError ('only bias = false supported rn')

    shard, shape, numel = parameter_sharding(linearlayer.weight, group=group)

    self.weightshard = nn.Parameter(shard)
    self.group = group
    self.shape = shape
    self.numel = numel

    self.in_features, self.out_features = linearlayer.in_features, linearlayer.out_features
  
  def forward(self, x):
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel, self.group)

    # x * W + b, except no b
    output = nn.functional.linear(input=x, weight=weights, bias = None)

    return output

class ShardedEmbedding(nn.Module):
  def __init__(self, embed: nn.Embedding, group=None):
    super().__init__()

    shard, shape, numel = parameter_sharding(embed.weight, group=group)

    self.weightshard = nn.Parameter(shard)
    self.group = group
    self.shape = shape
    self.numel = numel

  def forward(self, token_ids):
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel, self.group)
        
    return nn.functional.embedding(token_ids, weights)

class ShardedRMSNorm(nn.Module):
  def __init__(self, norm: nn.RMSNorm, group=None):
    super().__init__()

    shard, shape, numel = parameter_sharding(norm.weight, group=group)

    self.weightshard = nn.Parameter(shard)
    self.group = group
    self.shape = shape
    self.numel = numel

    self.normalized_shape = norm.normalized_shape
    self.eps = norm.eps

  def forward(self, x):
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel, self.group)

    return nn.functional.rms_norm(x, self.normalized_shape, weight=weights, eps=self.eps)

class ShardedExperts(nn.Module):
  def __init__(self, experts: nn.ModuleList, group=None):
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
    shard, shape, numel = parameter_sharding(combined_weights, group=group)

    self.weightshard = nn.Parameter(shard)
    self.group = group
    self.shape = shape
    self.numel = numel

  def forward(self, tokenstates, topkindices, topkweights):
    # Gather all expert weights once, across all ranks
    weights = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel, self.group)

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
    
def parameter_sharding(parameter_tensor, group=None):
  total_nodes = dist.get_world_size(group=group)

  # flatten it and split into chunks

  shape, numel = parameter_tensor.shape, parameter_tensor.numel()

  params = parameter_tensor.reshape(-1)

  shardsize = math.ceil(params.shape[0] / total_nodes)

  if params.numel() != shardsize * total_nodes:
    # pad with sufficient zeros

    amtmissing = shardsize * total_nodes - params.numel()
    params = nn.functional.pad(params, (0, amtmissing))

  params = params.split(shardsize)

  rank = dist.get_rank(group=group)

  myshard = params[rank].detach().clone()

  return myshard, shape, numel

def gather_params(myshard, shape, numel, group=None):
  # reconstruct the original tensor given shards

  total_nodes = dist.get_world_size(group=group)

  # thus we know the size of our tensor is going to be worldsize * shardsize

  allshards = [torch.empty_like(myshard) for i in range(total_nodes)]

  dist.all_gather(allshards, myshard, group=group)

  allshards = torch.cat(allshards, dim = 0)
  allshards = allshards[:numel]
  allshards = allshards.reshape(shape)

  return allshards

def reduce_scatter_gradients(gradients, group=None):
  total_nodes = dist.get_world_size(group=group)

  gradients = gradients.reshape(-1)

  shardsize = math.ceil(gradients.shape[0] / total_nodes)

  if gradients.numel() != shardsize * total_nodes:
    # pad with sufficient zeros

    amtmissing = shardsize * total_nodes - gradients.numel()
    gradients = nn.functional.pad(gradients, (0, amtmissing))

  output = torch.empty((shardsize,), dtype=gradients.dtype, device=gradients.device)

  gradients = gradients.contiguous() 

  dist.reduce_scatter_tensor(output=output, input=gradients, op=dist.ReduceOp.SUM, group=group)

  output.div_(total_nodes)

  return output

def shard_model(model, group=None):
  model.outputprojection = ShardedLinear(model.outputprojection, group=group)
  model.embeddings = ShardedEmbedding(model.embeddings, group=group)
  model.finalRMSnorm = ShardedRMSNorm(model.finalRMSnorm, group=group)

  for layer in model.transformerlayers:

    layer.attention.Wq = ShardedLinear(layer.attention.Wq, group=group)
    layer.attention.Wk = ShardedLinear(layer.attention.Wk, group=group)
    layer.attention.Wv = ShardedLinear(layer.attention.Wv, group=group)
    layer.attention.output = ShardedLinear(layer.attention.output, group=group)

    layer.firstnorm = ShardedRMSNorm(layer.firstnorm, group=group)
    layer.secondnorm = ShardedRMSNorm(layer.secondnorm, group=group)

    layer.attention.rmsNormQ = ShardedRMSNorm(layer.attention.rmsNormQ, group=group)
    layer.attention.rmsNormK = ShardedRMSNorm(layer.attention.rmsNormK, group=group)

    layer.moe.router.router = ShardedLinear(layer.moe.router.router, group=group)
    layer.moe.experts = ShardedExperts(layer.moe.experts, group=group)

  return model


# FSDP-shard the *local TP weight*, preserving the original TP collectives.
class ShardedTensorParallelLinear(nn.Module):
  def __init__(self, layer, group=None):
    from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism
    super().__init__()
    if not isinstance(layer, (ColumnwiseLinearParallelism, RowwiseLinearParallelism)):
      raise TypeError("expected a tensor-parallel linear")
    self.columnwise = isinstance(layer, ColumnwiseLinearParallelism)
    self.tp_group = layer.group
    self.fsdp_group = group
    self.shape = tuple(layer.weight.shape)
    self.numel = layer.weight.numel()
    shard, _, _ = parameter_sharding(layer.weight.detach(), group=group)
    self.weightshard = nn.Parameter(shard)
    self.tp_sharded = True

  def forward(self, x):
    from framework.tensorparallel import CopyToTensorParallelRegion, ReduceFromTensorParallelRegion
    weight = GatherShardedWeight.apply(self.weightshard, self.shape, self.numel, self.fsdp_group)
    if self.columnwise:
      x = CopyToTensorParallelRegion.apply(x, self.tp_group)
      return nn.functional.linear(x, weight)
    output = nn.functional.linear(x, weight)
    return ReduceFromTensorParallelRegion.apply(output, self.tp_group)


# Shard every local leaf parameter after PP/EP/TP partitioning.
# This is intentionally simple: each existing linear/embedding/norm is
# reconstructed for its forward call and reduce-scattered in backward.
# No custom flattening, optimizer state partitioning, or overlap.
def shard_local_modules(module, group=None):
  from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism

  for name, child in list(module.named_children()):
    if isinstance(child, (ColumnwiseLinearParallelism, RowwiseLinearParallelism)):
      replacement = ShardedTensorParallelLinear(child, group=group)
    elif isinstance(child, nn.Linear):
      replacement = ShardedLinear(child, group=group)
    elif isinstance(child, nn.Embedding):
      replacement = ShardedEmbedding(child, group=group)
    elif isinstance(child, nn.RMSNorm):
      replacement = ShardedRMSNorm(child, group=group)
    else:
      shard_local_modules(child, group=group)
      continue
    setattr(module, name, replacement)

  # Expert loops must enter all FSDP collectives in the same order,
  # including experts receiving no tokens on one data-parallel rank.
  if module.__class__.__name__ in ("MoE", "ExpertParallel"):
    module.fsdp_mode = True
  return module
