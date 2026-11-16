# Distributed Training Framework

A from-scratch Qwen3-MoE training framework with composable data parallelism
(DP), fully sharded data parallelism (FSDP), tensor parallelism (TP), pipeline
parallelism (PP / GPipe), context parallelism (CP), and expert parallelism (EP).

The `hybrid-mesh` branch provides a single configurable trainer and six-axis
communication mesh. The mesh dimensions are independent axes in this
implementation, and their product must equal the number of launched processes.
No separate training implementations are selected for individual combinations.

## Hybrid training

Run from the repository root:

```sh
# Four ranks, combining DP and TP.
torchrun --standalone --nproc-per-node=4 -m scripts.pretrain_hybrid \
  --synthetic --dp 2 --tp 2 --steps 3 \
  --optimizer adamw --lr 0.001 --weight-decay 0.01 \
  --scheduler cosine --warmup-steps 1 --decay-ratio 0.5 \
  --metrics-file outputs/metrics.jsonl \
  --checkpoint-dir outputs/checkpoints --save-every 1

# A different parallelism combination uses the exact same trainer:
torchrun --standalone --nproc-per-node=8 -m scripts.pretrain_hybrid \
  --synthetic --pp 2 --ep 2 --tp 2 --steps 2

# Resume the first run from its step-1 checkpoint, keeping optimizer,
# schedule, model, batch-size and mesh configuration identical:
torchrun --standalone --nproc-per-node=4 -m scripts.pretrain_hybrid \
  --synthetic --dp 2 --tp 2 --steps 3 \
  --optimizer adamw --lr 0.001 --weight-decay 0.01 \
  --scheduler cosine --warmup-steps 1 --decay-ratio 0.5 \
  --resume outputs/checkpoints/step-1 \
  --metrics-file outputs/metrics.jsonl
```

Use `--help` for full settings. All parallel degrees default to 1. The
available optimizers are AdamW, Adam, and SGD; schedules are constant, linear,
square-root decay, and cosine decay. Metrics include loss, gradient norm, LR,
step duration, tokens per second, and cumulative trained tokens. On rank 0,
`--metrics-file` appends one JSON record per optimizer update.

`--synthetic` uses reproducible generated tokens and needs no downloads.
Omit it to stream FineWeb with the Qwen3 tokenizer (requires `datasets` and
`transformers` plus dataset access).

**Checkpointing:** The hybrid trainer writes one model/optimizer/scheduler/RNG
state file per rank, along with mesh information and training counters. To
resume, use the same world size, mesh, architecture, and optimizer/scheduler
configuration. The total schedule length must remain fixed for nonconstant
schedules. The trainer recreates the FineWeb iterator and skips consumed
examples on resume; this is not a constant-time, exact stream-cursor restore,
and it assumes deterministic upstream sample ordering. Checkpoint directories
must be shared by participating ranks.

The hybrid training path selects NCCL on CUDA machines and Gloo otherwise.
Current CI runs on CPU/Gloo; multi-GPU/NCCL correctness and performance are
not yet validated. PP uses a simple GPipe forward-then-backward schedule.

## Correctness tests

```sh
torchrun --standalone --nproc-per-node=4 -m tests.test_hybrid_runtime --dp 2 --tp 2
torchrun --standalone --nproc-per-node=4 -m tests.test_hybrid_dense_reference --dp 2 --tp 2
torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode tp
```

GitHub Actions also tests several 2D/3D hybrid meshes, checkpoint resume, and
isolated communication groups. Small floating-point differences may occur
between distributed and dense reference execution.

## Legacy single-strategy runners

The original `scripts.pretrain`, `scripts.pretrain_tp`, and
`scripts.pretrain_pp` runners are retained for existing tests and
single-strategy experimentation. Prefer `scripts.pretrain_hybrid` for new jobs.

## Scope and limitations

This is a functional training-framework implementation, not a reproduction of
all of TorchFeather's performance features. It does not yet provide activation
checkpointing, multiple pipeline schedules, optimized GPU MoE kernels, compile
integration, automatic mixed precision in the hybrid trainer, or a
reshardable/world-size-independent checkpoint format. The model is fully
initialized before stage and parameter partitioning, which may temporarily
require much more memory than a production large-model implementation.

## Preparing a FineWeb sample

Install the data dependencies with `pip install datasets pyarrow`. The preparation script streams documents, normalizes and filters them, removes exact duplicates, and saves cleaned text in compressed Parquet shards. It does not create a train/test split or tokenize the text.

```sh
# Try 100 documents first.
python -m scripts.prepare_data --max-documents 100 --output-dir data/fineweb-smoke

# Prepare 100,000 documents (default size), writing up to 10,000 kept documents per shard.
python -m scripts.prepare_data --max-documents 100000 --output-dir data/fineweb-100k
```

The output contains `part-00000.parquet`, `part-00001.parquet`, etc., plus `manifest.json`. The manifest records source, seed, cleaning options, document counts and rejection reasons. The output directory must be new so existing datasets are never overwritten. Generated datasets under `data/` are ignored by Git.
