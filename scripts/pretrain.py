"""Configurable DDP or FSDP training for Qwen3-MoE.

Examples (CPU/Gloo):
    torchrun --standalone --nproc-per-node=2 -m scripts.pretrain --synthetic --parallelism fsdp
    torchrun --standalone --nproc-per-node=3 -m scripts.pretrain --synthetic --parallelism ddp --global-batch-size 6

Without --synthetic, stream FineWeb using the Qwen3 tokenizer.
"""
import argparse

import torch
from torch.utils.data import DataLoader

from framework.distributed import initialize_distributed, synchronize_model, cleanup_distributed
from framework.fsdp import shard_model
from framework.expertparallel import parallelize_ep
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.training.config import TrainingConfig
from framework.training.trainer import Trainer


def synthetic_batches(config, rank, vocab_size):
    gen = torch.Generator().manual_seed(config.seed + 1000 * rank)
    while True:
        yield torch.randint(
            vocab_size, (config.local_batch_size, config.seqlen + 1),
            generator=gen, dtype=torch.long,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", choices=("ddp", "fsdp", "ep"), default="fsdp")
    parser.add_argument("--synthetic", action="store_true", help="Train on random tokens without downloads")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=127, help="Used only with --synthetic")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--local-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--moe-inter-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-norm", type=float, default=1.0)
    options = parser.parse_args()

    rank, world_size, device = initialize_distributed()
    try:
        torch.set_num_threads(1)
        if options.steps < 1:
            raise ValueError("steps must be positive")
        if options.parallelism == "ep" and options.experts % world_size:
            raise ValueError("EP requires experts divisible by world size")
        if options.num_kv_heads < 1 or options.num_heads < 1 or options.num_heads % options.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if options.head_dim < 2 or options.head_dim % 2:
            raise ValueError("head_dim must be even and positive")
        if options.dim < 1 or options.moe_inter_dim < 1 or options.num_layers < 1 or options.seq_len < 1:
            raise ValueError("model dimensions, layers, and sequence length must be positive")
        if options.experts < 1 or not 1 <= options.top_k <= options.experts:
            raise ValueError("Require 1 <= top_k <= experts")
        if options.lr <= 0 or options.max_norm <= 0:
            raise ValueError("lr and max_norm must be positive")
        if options.synthetic and options.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        config = TrainingConfig(
            steps=options.steps,
            localbatchsize=options.local_batch_size,
            globalbatchsize=options.global_batch_size,
            seqlen=options.seq_len,
            learningrate=options.lr,
            maxnorm=options.max_norm,
            seed=42,
            warmupsteps=1,
            decayratio=0.5,
            decaytype="cosine",
            min_lr=0.1,
        )
        torch.manual_seed(config.seed)

        source = None
        if options.synthetic:
            vocab_size = options.vocab_size
            source = synthetic_batches(config, rank, vocab_size)
        else:
            from transformers import AutoTokenizer
            from framework.data.dataset import build_fineweb_dataset
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            vocab_size = len(tokenizer)
            source = DataLoader(
                build_fineweb_dataset(tokenizer, config.seqlen, rank, world_size),
                batch_size=config.local_batch_size,
                drop_last=True,
                num_workers=0,
            )

        args = Qwen3ModelArgs(
            vocab_size=vocab_size,
            max_seq_len=config.seqlen,
            dim=options.dim,
            moe_inter_dim=options.moe_inter_dim,
            numlayers=options.num_layers,
            numheads=options.num_heads,
            numkvheads=options.num_kv_heads,
            head_dim=options.head_dim,
            experts=options.experts,
            k=options.top_k,
        )
        model = Qwen3Model(args).to(device)
        synchronize_model(model)
        if options.parallelism == "fsdp":
            model = shard_model(model)
        elif options.parallelism == "ep":
            model = parallelize_ep(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        trainer = Trainer(model, optimizer, device, config, ep_mode=options.parallelism == "ep")
        trainer.train(source, config.steps)
        if rank == 0:
            print(f"PASS: {options.parallelism.upper()} training entry point completed", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
