"""Standalone GPipe trainer on CPU/Gloo.

Example (macOS):
    GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes=1 --nproc-per-node=2 \
      --rdzv-backend=static --master-addr=127.0.0.1 --master-port=29534 \
      -m scripts.pretrain_pp --synthetic --num-layers=4 --steps=2

Each rank owns one pipeline stage. The microbatch batch is shared across ranks;
the optimizer updates only local stage parameters.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from framework.distributed import initialize_distributed, cleanup_distributed
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.pipelineparallel import PipelineStage, gpipe_step
from framework.pp_checkpoint import save_pp_checkpoint, load_pp_checkpoint
from framework.training.config import TrainingConfig
from framework.training.lr_scheduler import build_lr_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use random tokens, no downloads")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--schedule-steps", type=int, default=None, help="Total LR schedule length, stable across resume")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=127, help="Synthetic vocabulary size")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--moe-inter-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Checkpoint step directory")
    options = parser.parse_args()

    # Validate before launching distributed collectives.
    if options.steps < 1: parser.error("--steps must be positive")
    if options.schedule_steps is not None and options.schedule_steps < options.steps:
        parser.error("--schedule-steps must be >= --steps")
    if options.batch_size < 1 or not 1 <= options.microbatches <= options.batch_size:
        parser.error("Require 1 <= microbatches <= batch-size")
    if options.seq_len < 1 or min(options.dim, options.moe_inter_dim, options.num_layers) < 1:
        parser.error("Model dimensions and sequence length must be positive")
    if options.num_kv_heads < 1 or options.num_heads < 1 or options.num_heads % options.num_kv_heads:
        parser.error("--num-heads must be divisible by --num-kv-heads")
    if options.head_dim < 2 or options.head_dim % 2:
        parser.error("--head-dim must be positive and even")
    if options.experts < 1 or not 1 <= options.top_k <= options.experts:
        parser.error("Require 1 <= top-k <= experts")
    if options.lr <= 0 or options.max_norm <= 0:
        parser.error("--lr and --max-norm must be positive")
    if options.synthetic and options.vocab_size < 2:
        parser.error("Synthetic vocab-size must be >= 2")
    if options.save_every < 0 or (options.save_every > 0 and not options.checkpoint_dir):
        parser.error("--save-every requires --checkpoint-dir and must be nonnegative")
    if options.resume and not options.synthetic:
        parser.error("Exact FineWeb stream resume is not implemented; use --synthetic for resume")

    rank, total_nodes, device = initialize_distributed()
    try:
        torch.set_num_threads(1)
        if options.num_layers < total_nodes:
            raise ValueError("Pipeline requires at least one transformer layer per rank")

        torch.manual_seed(options.seed)
        tokenizer = None
        if options.synthetic:
            vocab_size = options.vocab_size
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            vocab_size = len(tokenizer)

        args = Qwen3ModelArgs(
            vocab_size=vocab_size, max_seq_len=options.seq_len, dim=options.dim,
            moe_inter_dim=options.moe_inter_dim, numlayers=options.num_layers,
            numheads=options.num_heads, numkvheads=options.num_kv_heads,
            head_dim=options.head_dim, experts=options.experts, k=options.top_k,
        )
        model = Qwen3Model(args).to(device)
        stage = PipelineStage(model, rank, total_nodes)
        del model

        optimizer = torch.optim.AdamW(stage.parameters(), lr=options.lr)
        config = TrainingConfig(
            steps=options.schedule_steps or options.steps,
            localbatchsize=options.batch_size, globalbatchsize=options.batch_size,
            seqlen=options.seq_len, learningrate=options.lr,
            maxnorm=options.max_norm, seed=options.seed, warmupsteps=1,
            decayratio=0.5, decaytype="cosine", min_lr=0.1,
        )
        scheduler = build_lr_scheduler(optimizer, config)
        start_step = 0
        if options.resume:
            start_step = load_pp_checkpoint(
                options.resume, stage, optimizer, scheduler, rank, total_nodes, args
            )
        if start_step > options.steps:
            raise ValueError("Checkpoint step exceeds requested --steps")

        data_iterator = None
        if rank == 0 and not options.synthetic:
            from framework.data.dataset import build_fineweb_dataset
            dataset = build_fineweb_dataset(tokenizer, options.seq_len, 0, 1)
            data_iterator = iter(DataLoader(
                dataset, batch_size=options.batch_size, drop_last=True, num_workers=0,
            ))

        for step in range(start_step, options.steps):
            tokens = None
            if rank == 0:
                if options.synthetic:
                    generator = torch.Generator().manual_seed(options.seed + step)
                    tokens = torch.randint(
                        0, vocab_size, (options.batch_size, options.seq_len + 1),
                        generator=generator,
                    )
                else:
                    tokens = next(data_iterator)

            loss = gpipe_step(
                stage, tokens, optimizer, options.microbatches, rank,
                total_nodes, options.dim, device, max_norm=options.max_norm,
            )
            scheduler.step()
            if rank == total_nodes - 1:
                print(f"step={step + 1} loss={loss:.6f} lr={optimizer.param_groups[0]['lr']:.6g}", flush=True)

            if options.checkpoint_dir and options.save_every and (step + 1) % options.save_every == 0:
                save_pp_checkpoint(
                    options.checkpoint_dir, step + 1, stage,
                    optimizer, scheduler, rank, total_nodes, args,
                )

        if rank == 0: print("PASS: GPipe training entry point completed", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
