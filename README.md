# Distributed Training Framework

A PyTorch training framework for Qwen3-style Mixture-of-Experts models, supporting composable **data parallelism (DP), fully sharded data parallelism (FSDP), tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), and expert parallelism (EP)**.

A six-dimensional process-group mesh controls model partitioning, communication, and training. Parallelism degrees can be combined through one training entry point.

## Setup

Install PyTorch for your CPU or CUDA environment, then install the data dependencies:

```bash
pip install torch transformers datasets pyarrow
```

Run commands from the repository root. Distributed jobs use `torchrun` and require the product of the six parallelism degrees to equal the process count. Unspecified degrees default to 1.

## Training

Run a synthetic-data smoke test without downloading a dataset:

```bash
torchrun --standalone --nproc-per-node=4 -m scripts.pretrain_hybrid \
  --synthetic --dp 2 --tp 2 --steps 5
```

For real data, omit `--synthetic` to stream FineWeb or specify `--dataset-path` to train from prepared Parquet shards. Set model size, sequence length, batch sizes, and parallelism degrees through CLI options:

```bash
torchrun --standalone --nproc-per-node=4 -m scripts.pretrain_hybrid \
  --dp 2 --tp 2 --seq-len 128 --steps 100
```

Use `python -m scripts.pretrain_hybrid --help` for all options. The trainer supports gradient accumulation, GPipe microbatches, AdamW/Adam/SGD, learning-rate schedules, gradient clipping, and JSONL metrics.

## Preparing data

Prepare a cleaned, deduplicated FineWeb subset as compressed Parquet shards:

```bash
python -m scripts.prepare_data --output-dir data/fineweb-100k
```

Inspect document lengths, sampled token lengths, and cleaning statistics:

```bash
python -m scripts.inspect_data --dataset-path data/fineweb-100k
```

Train from the saved dataset:

```bash
torchrun --standalone --nproc-per-node=4 -m scripts.pretrain_hybrid \
  --dp 2 --tp 2 --dataset-path data/fineweb-100k --seq-len 128 --steps 100
```

The data loader tokenizes documents with the Qwen3 tokenizer, packs fixed-length next-token prediction examples, and assigns disjoint documents to data replicas. Model-parallel workers share each replica's batch.

## Checkpoints and metrics

Enable periodic per-rank checkpoints and training metrics with:

```bash
--checkpoint-dir outputs/checkpoints --save-every 10 --metrics-file outputs/metrics.jsonl
```

Resume using `--resume outputs/checkpoints/step-10` with the same world size, mesh, model, optimizer, and training configuration. Checkpoints preserve model, optimizer, scheduler, RNG state, and training counters; resuming a dataset stream replays and skips consumed samples.

## Tests

```bash
python -m unittest tests.test_cleaning tests.test_prepare_data tests.test_parquet_dataset
torchrun --standalone --nproc-per-node=4 -m tests.test_hybrid_dense_reference --dp 2 --tp 2
```

Additional distributed correctness tests run in GitHub Actions.

## Current scope

The framework prioritizes correctness and composability over performance. It uses GPipe for pipeline scheduling, initializes the dense model before partitioning, and currently validates distributed configurations primarily on CPU/Gloo; multi-GPU/NCCL performance is not yet established.
