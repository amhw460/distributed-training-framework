import os
import torch
from torch import nn
import torch.distributed as dist
from framework.fsdp import ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts

# distributed data paralellism (DDP)

def synchronize_gradients(model):
  total_nodes = dist.get_world_size()

  # FSDP shard gradients are already synchronized by reduce-scatter.
  sharded_parameter_ids = {id(module.weightshard) for module in model.modules()
    if isinstance(module, (ShardedLinear, ShardedEmbedding, ShardedRMSNorm))
  }

  for name, param in model.named_parameters():
    if id(param) in sharded_parameter_ids:
      continue

    has_gradient = False
    if param.grad is not None:
      has_gradient = True

    used = torch.tensor(1, device=param.device) if has_gradient else torch.tensor(0, device=param.device) 
    # some experts might have 0 gradient, we need to know which experts wer used

    dist.all_reduce(used, op=dist.ReduceOp.SUM)

    if used.item() == 0: continue # if no rank computed a gradient for this, then we know its not used, skpi

    gradient = param.grad if param.grad is not None else torch.zeros_like(param, device=param.device)

    dist.all_reduce(gradient, op=dist.ReduceOp.SUM)

    gradient.div_(total_nodes) # average, this is jus twhat they did in my class

    if param.grad is None:
      param.grad = gradient

def initialize_distributed():
  rank = int(os.environ["RANK"])
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  #initialize things

  dist.init_process_group(backend="gloo")
  device = torch.device("cpu")

  return rank, world_size, device

def cleanup_distributed(): 
  if dist.is_initialized(): dist.destroy_process_group()

def synchronize_model(model):
  
  with torch.no_grad():
    for name, param in model.named_parameters():
      # each param is a tensor with weights

      dist.broadcast(tensor = param, src=0) 

    for buffer in model.buffers():
      dist.broadcast(tensor=buffer, src = 0)


