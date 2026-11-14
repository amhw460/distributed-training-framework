"""Gradient clipping for models mixing replicated and manually sharded weights."""

import torch
import torch.distributed as dist

from framework.fsdp import ShardedLinear


def clip_hybrid_grad_norm_(model, max_norm):
    """Clip the global L2 gradient norm of a hybrid DDP/FSDP model.

    Replicated gradients have already been synchronized, so their squared norm
    must be counted once. Sharded gradients are different on each rank, so
    their squared norms must be summed across ranks.
    """
    sharded_parameter_ids = {
        id(module.weightshard)
        for module in model.modules()
        if isinstance(module, ShardedLinear)
    }

    # Preserve the ordinary single-device/DDP behavior when nothing is sharded.
    if not sharded_parameter_ids:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()

    parameters = list(model.parameters())
    if not parameters:
        return 0.0

    device = parameters[0].device
    replicated_norm_sq = torch.zeros((), dtype=torch.float64, device=device)
    sharded_norm_sq = torch.zeros((), dtype=torch.float64, device=device)

    for parameter in parameters:
        if parameter.grad is None:
            continue
        contribution = parameter.grad.detach().to(torch.float64).square().sum()
        if id(parameter) in sharded_parameter_ids:
            sharded_norm_sq += contribution
        else:
            replicated_norm_sq += contribution

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(sharded_norm_sq, op=dist.ReduceOp.SUM)

    global_norm = (replicated_norm_sq + sharded_norm_sq).sqrt()
    clip_coefficient = min(1.0, max_norm / (global_norm.item() + 1e-6))

    # Every rank uses the same coefficient for all its replicated and local
    # sharded gradients.
    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)

    return global_norm.item()
