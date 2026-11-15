"""Train Qwen3-MoE with two-rank tensor parallelism.

Real FineWeb training:
    torchrun --standalone --nproc-per-node=2 -m scripts.pretrain_tp

Offline smoke test:
    torchrun --standalone --nproc-per-node=2 -m scripts.pretrain_tp --synthetic --steps 2
"""

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


def shared_batches(rank, device, config, vocab_size, source=None, synthetic=False):
    """Every TP rank must receive exactly the same tokens and routing decisions."""
    source_iterator = iter(source) if rank == 0 and source is not None else None
    generator = torch.Generator().manual_seed(config.seed + 1)

    while True:
        if rank == 0:
            if synthetic:
                batch = torch.randint(
                    vocab_size,
                    (config.local_batch_size, config.seqlen + 1),
                    dtype=torch.long,
                    generator=generator,
                ).to(device)
            else:
                batch = next(source_iterator).to(device)
        else:
            batch = torch.empty(
                (config.local_batch_size, config.seqlen + 1),
                dtype=torch.long,
                device=device,
            )

        dist.broadcast(batch, src=0)
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use fake tokens; no downloads")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=127, help="Only for --synthetic")
    options = parser.parse_args()

    rank, world_size, device = initialize_distributed()
    try:
        if world_size != 2:
            raise ValueError("This TP training example expects exactly two ranks")
        if options.steps < 1:
            raise ValueError("--steps must be positive")
        if options.synthetic and options.vocab_size < 2:
            raise ValueError("--vocab-size must be at least 2")

        torch.set_num_threads(1)

        config = TrainingConfig(
            steps=options.steps,
            localbatchsize=2,
            globalbatchsize=8,
            seqlen=16,
            learningrate=1e-3,
            maxnorm=1.0,
            seed=42,
            warmupsteps=1,
            decayratio=0.5,
            decaytype="cosine",
            min_lr=0.1,
        )
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
            source = DataLoader(
                build_fineweb_dataset(tokenizer, config.seqlen, rank=0, world_size=1),
                batch_size=config.local_batch_size,
                drop_last=True,
                num_workers=0,
            )

        # All TP ranks construct the same model, including the vocabulary size.
        vocab_size_tensor = torch.tensor([vocab_size], dtype=torch.long, device=device)
        dist.broadcast(vocab_size_tensor, src=0)
        vocab_size = int(vocab_size_tensor.item())

        args = Qwen3ModelArgs(
            vocab_size=vocab_size,
            max_seq_len=config.seqlen,
            dim=32,
            moe_inter_dim=64,
            numlayers=2,
            numheads=4,
            numkvheads=2,
            head_dim=8,
            experts=4,
            k=2,
        )

        model = Qwen3Model(args).to(device)
        synchronize_model(model)
        model = paralellize_model(model)  # TP, not FSDP

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        trainer = Trainer(model, optimizer, device, config, tp_mode=True)

        batches = shared_batches(
            rank, device, config, vocab_size, source=source, synthetic=options.synthetic,
        )
        trainer.train(batches, config.steps)
        if rank == 0:
            print("PASS: TP training script completed", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
