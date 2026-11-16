import os

import torch
import torch.distributed as dist

from framework.fsdp import ShardedEmbedding, ShardedExperts, ShardedLinear, ShardedRMSNorm


def synchronize_gradients(model, group=None):
    total_nodes = dist.get_world_size(group=group)

    # fsdp already syncs its own shards through reduce-scatter
    sharded_parameter_ids = {
        id(module.weightshard)
        for module in model.modules()
        if isinstance(
            module, (ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts)
        )
    }

    for name, param in model.named_parameters():
        if id(param) in sharded_parameter_ids:
            continue

        # some experts might not get any tokens on a rank
        has_gradient = param.grad is not None
        used = torch.tensor(int(has_gradient), device=param.device)
        dist.all_reduce(used, op=dist.ReduceOp.SUM, group=group)

        if used.item() == 0:
            continue

        gradient = (
            param.grad if param.grad is not None
            else torch.zeros_like(param, device=param.device)
        )
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=group)
        gradient.div_(total_nodes)

        if param.grad is None:
            param.grad = gradient


def initialize_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo")
    device = torch.device("cpu")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def synchronize_model(model, group=None):
    source = dist.get_global_rank(group, 0) if group is not None else 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            dist.broadcast(tensor=param, src=source, group=group)

        for buffer in model.buffers():
            dist.broadcast(tensor=buffer, src=source, group=group)
