"""Two-rank correctness checks for manual DDP.

Run from the repository root:
  torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \
    --master-addr=127.0.0.1 --master-port=29501 -m tests.test_distributed

No FineWeb download or GPU is required.
"""

import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist

from framework.distributed import (
    cleanup_distributed,
    initialize_distributed,
    synchronize_gradients,
    synchronize_model,
)
from framework.training.checkpoint import Checkpoint
from framework.training.config import TrainingConfig
from framework.training.trainer import Trainer


class TinyMoE(nn.Module):
    """A tiny language model with one expert per rank to exercise missing gradients."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.experts = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])
        self.projection = nn.Linear(8, 32, bias=False)

    def forward(self, tokens):
        # Rank-specific input tokens route to different experts.
        expert_id = int(tokens[0, 0].item())
        activations = self.experts[expert_id](self.embedding(tokens))
        return self.projection(torch.tanh(activations))


class DeterministicBatches:
    """Checkpointable rank-specific batches without an external dataset."""

    def __init__(self, rank):
        self.rank = rank
        self.next_batch = 0

    def __iter__(self):
        while True:
            generator = torch.Generator().manual_seed(1000 + 10000 * self.rank + self.next_batch)
            batch = torch.randint(2, 32, (1, 5), generator=generator)
            batch[0, 0] = self.rank
            self.next_batch += 1
            yield batch

    def state(self):
        return self.next_batch

    def load_state(self, state):
        self.next_batch = state


def assert_identical_replicas(model, world_size):
    for name, parameter in model.named_parameters():
        values = parameter.detach().contiguous()
        gathered = [torch.empty_like(values) for _ in range(world_size)]
        dist.all_gather(gathered, values)
        for other in gathered[1:]:
            torch.testing.assert_close(gathered[0], other, rtol=0, atol=0, msg=f"Replica mismatch: {name}")


def test_missing_expert_gradients(rank):
    model = nn.Module()
    for name in ("shared", "expert_a", "expert_b", "never_used"):
        model.register_parameter(name, nn.Parameter(torch.tensor(0.0)))

    model.shared.grad = torch.tensor(2.0 if rank == 0 else 6.0)
    if rank == 0:
        model.expert_a.grad = torch.tensor(4.0)
    else:
        model.expert_b.grad = torch.tensor(10.0)

    synchronize_gradients(model)
    assert model.shared.grad.item() == 4.0
    assert model.expert_a.grad.item() == 2.0
    assert model.expert_b.grad.item() == 5.0
    assert model.never_used.grad is None


def make_training_state(config, rank, folder):
    torch.manual_seed(12345 + rank)
    model = TinyMoE()
    synchronize_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    trainer = Trainer(model, optimizer, "cpu", config)
    batches = DeterministicBatches(rank)
    checkpoint = Checkpoint(trainer, folder, batches)
    return model, trainer, batches, checkpoint


def test_train_and_resume(rank, world_size):
    config = TrainingConfig(
        steps=4, localbatchsize=1, globalbatchsize=4, seqlen=4,
        learningrate=1e-3, maxnorm=1.0, seed=12345,
        warmupsteps=1, decayratio=0.5, decaytype="cosine", min_lr=0.1,
    )

    shared_path = [tempfile.mkdtemp(prefix="manual-ddp-") if rank == 0 else None]
    dist.broadcast_object_list(shared_path, src=0)
    folder = Path(shared_path[0])

    try:
        # Reference: four uninterrupted optimizer updates.
        baseline_model, baseline, baseline_batches, _ = make_training_state(config, rank, folder)
        assert baseline.steps_gradient_accumulation == 2
        baseline.train(baseline_batches, 4)
        assert_identical_replicas(baseline_model, world_size)

        # Save halfway through, then reload into freshly constructed objects.
        _, partial, partial_batches, saver = make_training_state(config, rank, folder)
        partial.train(partial_batches, 2)
        saver.save(partial.trainingstep)  # Only rank 0 should write.
        dist.barrier()  # No rank should load before rank 0 has finished saving.
        assert (folder / "step-2.pt").is_file()

        resumed_model, resumed, resumed_batches, loader = make_training_state(config, rank, folder)
        loader.load(folder / "step-2.pt")
        assert resumed.trainingstep == 2
        assert resumed_batches.state() == 4
        resumed.train(resumed_batches, config.steps - resumed.trainingstep)
        assert_identical_replicas(resumed_model, world_size)

        for name, expected in baseline_model.state_dict().items():
            actual = resumed_model.state_dict()[name]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"Resume mismatch: {name}")

        torch.testing.assert_close(
            resumed.optimizer.state_dict(), baseline.optimizer.state_dict(), rtol=0, atol=0,
        )
        assert resumed.scheduler.last_epoch == baseline.scheduler.last_epoch
        assert resumed.state() == baseline.state()
        assert resumed_batches.state() == baseline_batches.state()
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(folder)
        dist.barrier()


def main():
    rank, world_size, device = initialize_distributed()
    try:
        assert device.type == "cpu", "This test expects the CPU/Gloo implementation"
        assert world_size == 2, "Launch this test with exactly 2 ranks"
        test_missing_expert_gradients(rank)
        if rank == 0:
            print("PASS: Gradient all-reduce handles missing MoE expert gradients", flush=True)
        test_train_and_resume(rank, world_size)
        if rank == 0:
            print("PASS: Replicas match; interrupted training resumes identically", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
