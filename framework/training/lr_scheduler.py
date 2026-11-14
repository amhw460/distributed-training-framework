import math
from torch.optim.lr_scheduler import LambdaLR
from framework.training.config import TrainingConfig


def build_lr_scheduler(optimizer, config: TrainingConfig):
    steps = config.steps
    warmup_steps = config.warmup_steps
    decay_steps = round(steps * config.decay_ratio) if config.decay_ratio is not None else steps - warmup_steps
    stable_steps = steps + 1 - warmup_steps - decay_steps

    if not (0 <= warmup_steps <= steps and 0 < decay_steps <= steps + 1 - warmup_steps):
        raise ValueError("warmup and decay steps must fit within the schedule")
    if config.decay_type not in ("linear", "sqrt", "cosine"):
        raise ValueError(f"unsupported decay type: {config.decay_type}")

    def step_lr(current_step):
        if current_step < warmup_steps:
            return (current_step + 1) / warmup_steps
        if current_step < warmup_steps + stable_steps:
            return 1.0

        progress = min(1.0, (current_step - warmup_steps - stable_steps + 1) / decay_steps)
        if config.decay_type == "linear":
            factor = 1 - progress
        elif config.decay_type == "sqrt":
            factor = 1 - math.sqrt(progress)
        else:
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return config.min_lr + (1 - config.min_lr) * factor

    return LambdaLR(optimizer, step_lr)
