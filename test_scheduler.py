
import math
import torch

from trainingconfig import TrainingConfig
from lr_scheduler import build_lr_scheduler

for decay_type in ["linear", "sqrt", "cosine"]:
    config = TrainingConfig(
        steps=20,
        localbatchsize=2,
        globalbatchsize=8,
        seqlen=8,
        learningrate=0.001,
        maxnorm=1.0,
        seed=42,
        warmupsteps=4,
        decayratio=0.5,
        decaytype=decay_type,
        min_lr=0.1,
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    scheduler = build_lr_scheduler(optimizer, config)

    observed = []

    # Record the initial LR, then advance 20 updates.
    for step in range(config.steps + 1):
        observed.append(optimizer.param_groups[0]["lr"])

        if step < config.steps:
            optimizer.zero_grad()
            parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            scheduler.step()

    base_lr = 0.001
    warmup = config.warmup_steps
    decay = round(config.steps * config.decay_ratio)
    stable = config.steps + 1 - warmup - decay

    expected = []
    for step in range(config.steps + 1):
        if step < warmup:
            factor = (step + 1) / warmup
        elif step < warmup + stable:
            factor = 1.0
        else:
            progress = (step - warmup - stable + 1) / decay

            if decay_type == "linear":
                factor = 1 - progress
            elif decay_type == "sqrt":
                factor = 1 - math.sqrt(progress)
            else:
                factor = 0.5 * (1 + math.cos(math.pi * progress))

            factor = config.min_lr + (1 - config.min_lr) * factor

        expected.append(base_lr * factor)

    for step, (actual, target) in enumerate(zip(observed, expected)):
        assert math.isclose(actual, target, abs_tol=1e-10), (
            f"{decay_type}, step {step}: got {actual}, expected {target}"
        )

    assert math.isclose(observed[-1], 0.0001, abs_tol=1e-10)
    print(f"PASS: {decay_type} schedule")

print("ALL SCHEDULER TESTS PASSED")
