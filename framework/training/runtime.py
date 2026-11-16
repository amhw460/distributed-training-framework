"""Runtime optimizer/scheduler factories for composable hybrid training."""

from dataclasses import dataclass

import torch

from framework.training.lr_scheduler import build_lr_scheduler


@dataclass(frozen=True)
class ScheduleConfig:
    steps: int
    warmup_steps: int = 0
    decay_ratio: float | None = None
    decay_type: str = "cosine"
    min_lr: float = 0.0


def build_optimizer(parameters, *, name="adamw", lr=1e-3, weight_decay=0.0,
                    beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.0):
    """Build an optimizer *after* model transformations select local parameters."""
    if lr <= 0 or weight_decay < 0 or eps <= 0:
        raise ValueError("Invalid optimizer learning rate, weight decay, or epsilon")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError("Adam betas must be in [0, 1)")
    if not (0 <= momentum < 1):
        raise ValueError("SGD momentum must be in [0, 1)")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, betas=(beta1, beta2),
                                 eps=eps, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, betas=(beta1, beta2),
                                eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_hybrid_scheduler(optimizer, *, name="constant", steps=1, warmup_steps=0,
                           decay_ratio=None, min_lr_factor=0.0):
    """The initial optimizer LR applies to step 1; step() advances after an update."""
    if name == "constant":
        if warmup_steps or decay_ratio is not None:
            raise ValueError("Warmup/decay requires a non-constant scheduler")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if name not in ("linear", "sqrt", "cosine"):
        raise ValueError(f"Unsupported learning-rate schedule: {name}")
    if not 0 <= min_lr_factor <= 1:
        raise ValueError("min_lr_factor must be in [0, 1]")
    config = ScheduleConfig(steps=steps, warmup_steps=warmup_steps,
                            decay_ratio=decay_ratio, decay_type=name,
                            min_lr=min_lr_factor)
    return build_lr_scheduler(optimizer, config)
