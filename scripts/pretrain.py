
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.training.config import TrainingConfig
from framework.data.dataset import build_fineweb_dataset
from framework.training.trainer import Trainer
from framework.training.checkpoint import Checkpoint



from framework.distributed import (
    initialize_distributed,
    synchronize_model,
    cleanup_distributed,
)


def main():
    # Training configuration

    rank, world_size, device = initialize_distributed()

    config = TrainingConfig(
        steps=50,
        localbatchsize=2,
        globalbatchsize=8,
        seqlen=16,
        learningrate=1e-3,
        maxnorm=1.0,
        seed=42,
        warmupsteps=5,
        decayratio=0.5,
        decaytype="cosine",
        min_lr=0.1,
    )

    torch.manual_seed(config.seed)

    # Dataset
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    dataset = build_fineweb_dataset(
        tokenizer,
        config.seqlen,
        rank,
        world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.local_batch_size,
        drop_last=True,
        num_workers=0,
    )

    # Small Qwen3-MoE model
    args = Qwen3ModelArgs(
        vocab_size=len(tokenizer),
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

    model = Qwen3Model(args)

    model = model.to(device)
    synchronize_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr
    )

    trainer = Trainer(model, optimizer, device, config)
    checkpoint = Checkpoint(trainer, "checkpoints", dataset)

    # Resume if a checkpoint path was supplied
    if len(sys.argv) > 1:
        checkpoint.load(sys.argv[1])

    # Train until the configured total step count
    remaining = config.steps - trainer.trainingstep

    if remaining > 0:
        trainer.train(
            loader,
            remaining,
            checkpoint=checkpoint,
            save_every=10,
        )
        checkpoint.save(trainer.trainingstep)

    cleanup_distributed()

if __name__ == "__main__":
    main()
