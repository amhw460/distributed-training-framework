import torch
import torch.distributed as dist
from framework.fsdp import ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts
from framework.tensorparallel import ColumnwiseLinearParallelism, RowwiseLinearParallelism

def clip_hybrid_grad_norm_(model, max_norm, group=None):

    # hybrid gradient clipping for fsdp
    
    sharded_parameter_ids = {id(module.weightshard) for module in model.modules() if 
    isinstance(module,(ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts))}

    # if its ddp just use regular thing
    if not sharded_parameter_ids:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

    parameters = list(model.parameters())
    if not parameters:
        return 0.0

    device = parameters[0].device
    replicated_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    sharded_norm_sq = torch.zeros((), dtype=torch.float64, device=device)

    for parameter in parameters:
        if parameter.grad is None: continue

        contribution = parameter.grad.detach().to(torch.float64).square().sum()
        if id(parameter) in sharded_parameter_ids:
            sharded_norm_sq += contribution
        else:
            replicated_norm_sq += contribution

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sharded_norm_sq, op=dist.ReduceOp.SUM, group=group)

    global_norm = (replicated_norm_sq + sharded_norm_sq).sqrt()
    clip_coefficient = min(1.0, max_norm / (global_norm.item() + 1e-6))

    # scaling coefficient is same

    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)

    return global_norm.item()


def clip_tp_grad_norm_(model, max_norm, group=None):
    sharded_types = (ColumnwiseLinearParallelism, RowwiseLinearParallelism)

    sharded_parameter_ids = {id(module.weight) for module in model.modules() if isinstance(module, sharded_types)}

    parameters = list(model.parameters())
    if not parameters:
        return 0.0

    device = parameters[0].device

    sharded_norm_sq = torch.zeros((), dtype=torch.float64, device=device)

    replicated_norm_sq = torch.zeros((), dtype=torch.float64, device=device)

    for parameter in parameters:
        if parameter.grad is None:
            continue

        contribution = (parameter.grad.detach().to(torch.float64).square().sum())

        if id(parameter) in sharded_parameter_ids:
            sharded_norm_sq += contribution
        else:
            replicated_norm_sq += contribution

    # Combine contributions from different TP weight shards
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sharded_norm_sq, op=dist.ReduceOp.SUM, group=group)

    # Replicated parameters should only be counted once
    global_norm = (sharded_norm_sq + replicated_norm_sq).sqrt()

    clip_coefficient = min(1.0, max_norm / (global_norm.item() + 1e-6))

    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)

    return global_norm.item()

def clip_pp_grad_norm_(stage, max_norm, group=None):
    """Clip disjoint pipeline-stage gradients using one global norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")

    params = list(stage.parameters())
    if not params:
        raise ValueError("pipeline stage has no parameters")

    local_sq = torch.zeros((), dtype=torch.float64, device=params[0].device)
    for param in params:
        if param.grad is not None:
            local_sq += param.grad.detach().to(torch.float64).square().sum()

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=group)

    global_norm = local_sq.sqrt().item()
    coefficient = min(1.0, max_norm / (global_norm + 1e-6))

    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.grad.mul_(coefficient)

    return global_norm
