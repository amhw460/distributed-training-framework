# Train Qwen3-MoE with configurable tensor parallelism.
# Real FineWeb training:
# torchrun --standalone --nproc-per-node=2 -m scripts.pretrain_tp
# Offline smoke test:
# torchrun --standalone --nproc-per-node=2 -m scripts.pretrain_tp --synthetic --steps 2

import argparse

import torch
import torch.distributed as dist

from framework.distributed import (
    initialize_distributed,
    synchronize_model,
    cleanup_distributed,
)
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.tensorparallel import paralellize_model
from framework.training.config import TrainingConfig
from framework.training.trainer import Trainer


# Every TP rank must receive exactly the same tokens and routing decisions.
def shared_batches(rank, device, config, vocab_size, source=None, synthetic=False):
    source_iterator = iter(source) if rank == 0 and source is not None else None
    generator = torch.Generator().manual_seed(config.seed + 1)

    while True:
        if rank == 0:
            if synthetic:
                batch = torch.randint(vocab_size, (config.local_batch_size, config.seqlen + 1), dtype=torch.long, generator=generator).to(device)
            else:
                batch = next(source_iterator).to(device)
        else:
            batch = torch.empty((config.local_batch_size, config.seqlen + 1), dtype=torch.long, device=device)

        dist.broadcast(batch, src=0)
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use fake tokens; no downloads")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=127, help="Only for --synthetic")
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
        if (
            options.num_heads < world_size
            or options.num_kv_heads < world_size
            or options.num_heads % world_size
            or options.num_kv_heads % world_size
        ):
            raise ValueError("Both Q and KV head counts must be divisible by TP world size")
        if options.num_heads % options.num_kv_heads:
            raise ValueError("Q head count must be divisible by KV head count")
        if options.head_dim < 2 or options.head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if options.experts < 1 or not 1 <= options.top_k <= options.experts:
            raise ValueError("Require 1 <= top_k <= experts")
        if options.moe_inter_dim < world_size:
            raise ValueError("moe_inter_dim must be at least world size")
        if options.dim < 1 or options.num_layers < 1 or options.seq_len < 1:
            raise ValueError("Model dimensions, layers and sequence length must be positive")
        if options.lr <= 0 or options.max_norm <= 0:
            raise ValueError("lr and max_norm must be positive")
        if options.steps < 1:
            raise ValueError("--steps must be positive")
        if options.synthetic and options.vocab_size < 2:
            raise ValueError("--vocab-size must be at least 2")

        torch.set_num_threads(1)

        config = TrainingConfig(steps=options.steps, localbatchsize=options.local_batch_size, globalbatchsize=options.global_batch_size, seqlen=options.seq_len, learningrate=options.lr, maxnorm=options.max_norm, seed=42, warmupsteps=1, decayratio=0.5, decaytype="cosine", min_lr=0.1)
        torch.manual_seed(config.seed)

        # FineWeb is read only on rank 0. A single source feeds both TP ranks.
        source = None
        vocab_size = options.vocab_size if options.synthetic else 0
        if rank == 0 and not options.synthetic:
            from torch.utils.data import DataLoader
            from transformers import AutoTokenizer
            from framework.data.dataset import build_fineweb_dataset

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            vocab_size = len(tokenizer)
            source = DataLoader(build_fineweb_dataset(tokenizer, config.seqlen, rank=0, world_size=1), batch_size=config.local_batch_size, drop_last=True, num_workers=0)

        # All TP ranks construct the same model, including the vocabulary size.
        vocab_size_tensor = torch.tensor([vocab_size], dtype=torch.long, device=device)
        dist.broadcast(vocab_size_tensor, src=0)
        vocab_size = int(vocab_size_tensor.item())

        args = Qwen3ModelArgs(vocab_size=vocab_size, max_seq_len=config.seqlen, dim=options.dim, moe_inter_dim=options.moe_inter_dim, numlayers=options.num_layers, numheads=options.num_heads, numkvheads=options.num_kv_heads, head_dim=options.head_dim, experts=options.experts, k=options.top_k)

        model = Qwen3Model(args).to(device)
        synchronize_model(model)
        model = paralellize_model(model)  # TP, not FSDP

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        trainer = Trainer(model, optimizer, device, config, tp_mode=True)

        batches = shared_batches(rank, device, config, vocab_size, source=source, synthetic=options.synthetic)
        trainer.train(batches, config.steps)
        if rank == 0:
            print("PASS: TP training script completed", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
