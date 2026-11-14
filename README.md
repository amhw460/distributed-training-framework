# Distributed Training Framework

Building a distributed training framework from scratch using Qwen3-MoE, in the style of TorchTitan.

## Layout

```text
framework/
  models/          # Qwen3 model, model arguments, and mixture of experts
  training/        # Trainer, checkpointing, training config, and LR scheduler
  data/            # Text dataset and FineWeb dataset construction
  distributed.py   # Distributed training placeholder
scripts/
  pretrain.py      # FineWeb training and checkpoint resume entry point
  run.py           # Original small-model demo
tests/
  test_model.py
  test_accumulation.py
  test_scheduler.py
```

## Run

Run these commands from the repository root using the existing virtual environment.
FineWeb training downloads the tokenizer and streams the dataset.

```sh
.venv/bin/python -m scripts.pretrain
.venv/bin/python -m scripts.pretrain checkpoints/step-10.pt
```

## Tests

Run all existing checks:

```sh
.venv/bin/python -m tests
```

Or run an individual check:

```sh
.venv/bin/python -m tests.test_model
.venv/bin/python -m tests.test_accumulation
.venv/bin/python -m tests.test_scheduler
```
