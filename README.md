# Distributed Training Framework

A from-scratch Qwen3-MoE training project with manual DDP, FSDP, Tensor Parallelism (TP), and GPipe Pipeline Parallelism (PP). Models and training configuration are parameterized; execution currently uses **CPU/Gloo**. EP and combined multi-dimensional parallelism are not implemented yet.

## Training

From the repository root, using your Python environment:

```sh
# CPU-only offline smoke tests (no tokenizer or FineWeb download)
torchrun --standalone --nproc-per-node=2 -m scripts.pretrain --parallelism ddp --synthetic --steps 2
torchrun --standalone --nproc-per-node=2 -m scripts.pretrain --parallelism fsdp --synthetic --steps 2
torchrun --standalone --nproc-per-node=2 -m scripts.pretrain_tp --synthetic --steps 2

# Use different rank counts and model/batch configurations
torchrun --standalone --nproc-per-node=3 -m scripts.pretrain_tp --synthetic \
  --num-heads 6 --num-kv-heads 3 --head-dim 4 --moe-inter-dim 14 \
  --num-layers 3 --experts 5 --top-k 3 \
  --local-batch-size 3 --global-batch-size 9 --seq-len 7
```

Omit `--synthetic` to stream FineWeb and use the Qwen3 tokenizer. `--help` lists available model and training settings. TP requires both Q and KV head counts to be divisible by the number of TP ranks; it also requires the same token batch on every rank (the TP runner broadcasts from rank 0). DDP/FSDP use different data per rank.


### GPipe pipeline training

On macOS, specify the loopback Gloo interface and an IPv4 rendezvous address:

```sh
export GLOO_SOCKET_IFNAME=lo0
torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29534 \
  -m scripts.pretrain_pp --synthetic --steps 2 --schedule-steps 4 \
  --num-layers 3 --batch-size 5 --microbatches 2 \
  --checkpoint-dir checkpoints/pp --save-every 2

# Resume from step 2, keeping the same LR schedule and model configuration:
torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \
  --master-addr=127.0.0.1 --master-port=29535 \
  -m scripts.pretrain_pp --synthetic --steps 4 --schedule-steps 4 \
  --num-layers 3 --batch-size 5 --microbatches 2 \
  --resume checkpoints/pp/step-2
```

The trainer uses GPipe microbatch accumulation, global gradient clipping across disjoint pipeline stages, and rank-local model/optimizer/scheduler checkpoints. The `--schedule-steps` setting must remain unchanged across resume. Non-synthetic FineWeb streaming works, but exact data-stream resumption is not supported. PP currently constructs a full model before selecting each rank's layers; it is not yet memory-scalable during initialization.

## Tests

```sh
python -m tests
torchrun --standalone --nproc-per-node=2 -m tests.test_full_fsdp
torchrun --standalone --nproc-per-node=2 -m tests.test_full_tp
torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode tp
torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode fsdp
torchrun --standalone --nproc-per-node=3 -m tests.test_parameter_variations --mode ddp
```

CI also exercises TP with 1 and 4 ranks, PP with 1/2/3 ranks, and configurable multi-rank training entry points.

## Current limitations

- CUDA/NCCL GPU execution is not wired up; the distributed initializer uses CPU/Gloo.
- FSDP and TP checkpoint/resume are not implemented; do not attempt to save partial rank-local states as complete checkpoints.
- PP checkpoints are rank-local and require an identical pipeline partition on resume; FineWeb data-stream state is not restored.
- Manual DDP uses replicated model parameters and explicit gradient synchronization.
- Large-model scaling, performance, and all combinations of valid hyperparameters have not been benchmarked or validated.
