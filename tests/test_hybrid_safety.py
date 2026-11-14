"""Two-rank checks for hybrid DDP gradient synchronization and checkpoint safety.

Run:
    torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \\
        --master-addr=127.0.0.1 --master-port=29504 -m tests.test_hybrid_safety
"""

import tempfile

import torch
import torch.distributed as dist
from torch import nn

from framework.distributed import synchronize_gradients
from framework.fsdp import ShardedLinear
from framework.training.checkpoint import Checkpoint


class SmallHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.replicated = nn.Parameter(torch.tensor([0.0]))
        self.outputprojection = ShardedLinear(nn.Linear(1, 2, bias=False))


class MinimalTrainer:
    def __init__(self, model):
        self.model = model


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        rank = dist.get_rank()
        torch.manual_seed(100)
        model = SmallHybrid()

        model.replicated.grad = torch.tensor([2.0 if rank == 0 else 6.0])
        shard_grad = torch.full_like(
            model.outputprojection.weightshard, 10.0 if rank == 0 else 20.0
        )
        model.outputprojection.weightshard.grad = shard_grad.clone()

        synchronize_gradients(model)
        assert model.replicated.grad.item() == 4.0
        torch.testing.assert_close(model.outputprojection.weightshard.grad, shard_grad)

        checkpoint = Checkpoint(MinimalTrainer(model), tempfile.gettempdir(), None)
        for operation in (lambda: checkpoint.save(1), lambda: checkpoint.load("missing.pt")):
            try:
                operation()
            except RuntimeError as error:
                assert "ShardedLinear" in str(error)
            else:
                raise AssertionError("Checkpoint operation should reject sharded models")

        if rank == 0:
            print("PASS: replicated gradients synchronize; shards stay independent", flush=True)
            print("PASS: sharded checkpoint save/load rejected", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
