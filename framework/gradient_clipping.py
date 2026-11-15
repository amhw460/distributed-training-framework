import torch
import torch.distributed as dist
from framework.fsdp import ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts


def clip_hybrid_grad_norm_(model, max_norm):

    # hybrid gradient clipping for fsdp
    
    sharded_parameter_ids = {id(module.weightshard) for module in model.modules() if isinstance(module, (ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts))}

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
        dist.all_reduce(sharded_norm_sq, op=dist.ReduceOp.SUM)

    global_norm = (replicated_norm_sq + sharded_norm_sq).sqrt()
    clip_coefficient = min(1.0, max_norm / (global_norm.item() + 1e-6))

    # scaling coefficient is same

    with torch.no_grad():
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip_coefficient)

    return global_norm.item()
