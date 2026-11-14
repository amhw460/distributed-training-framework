"""Verify a manually sharded linear layer against ordinary dense training.

Run from the repository root:
    torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \
        --master-addr=127.0.0.1 --master-port=29502 -m tests.test_fsdp

No GPU, FineWeb download, or pretraining runner is required.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from framework.fsdp import ShardedLinear, gather_params


def test_sharded_linear(rank, world_size):
    # Use a 3x5 weight: 15 values cannot be split evenly across two ranks.
    # Start with different seeds to verify rank-0 parameter broadcasting.
    torch.manual_seed(41 + rank)
    dense = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        dist.broadcast(dense.weight, src=0)
    initial_weight = dense.weight.detach().clone()

    sharded = ShardedLinear(dense)
    shard_size = math.ceil(dense.weight.numel() / world_size)
    assert sharded.weightshard.numel() == shard_size
    assert sum(p.numel() for p in sharded.parameters()) == shard_size, (
        "ShardedLinear should only register its owned shard, not the original full layer"
    )

    # Equal local batch sizes make the average of rank losses equal to the
    # loss on the combined global batch.
    generator = torch.Generator().manual_seed(981 + rank)
    x = torch.randn(2, 5, generator=generator)
    target = torch.randn(2, 3, generator=generator)

    all_x = [torch.empty_like(x) for _ in range(world_size)]
    all_target = [torch.empty_like(target) for _ in range(world_size)]
    dist.all_gather(all_x, x)
    dist.all_gather(all_target, target)

    # Check forward pass against the original dense weight.
    actual_output = sharded(x)
    expected_output = F.linear(x, initial_weight)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)

    # Compute the reference global-batch gradient with a normal linear layer.
    reference = nn.Linear(5, 3, bias=False)
    with torch.no_grad():
        reference.weight.copy_(initial_weight)

    reference_loss = F.mse_loss(
        reference(torch.cat(all_x, dim=0)),
        torch.cat(all_target, dim=0),
    )
    reference_loss.backward()

    # Each rank computes its own local loss; the custom autograd backward
    # must reduce-scatter the averaged gradient to its owned shard.
    local_loss = F.mse_loss(actual_output, target)
    local_loss.backward()

    total_elements = reference.weight.numel()
    padded_elements = shard_size * world_size - total_elements
    expected_full_gradient = F.pad(
        reference.weight.grad.reshape(-1), (0, padded_elements)
    )
    expected_shard_gradient = expected_full_gradient.chunk(world_size)[rank]
    torch.testing.assert_close(
        sharded.weightshard.grad,
        expected_shard_gradient,
        rtol=1e-5,
        atol=1e-6,
    )

    # Update sharded and dense weights with identical AdamW settings.
    # After gathering updated weight shards, both results should agree.
    local_optimizer = torch.optim.AdamW(
        sharded.parameters(), lr=0.01, weight_decay=0.0
    )
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=0.01, weight_decay=0.0
    )
    local_optimizer.step()
    reference_optimizer.step()

    reconstructed_weight = gather_params(
        sharded.weightshard.detach(), sharded.shape, sharded.numel
    )
    torch.testing.assert_close(
        reconstructed_weight,
        reference.weight.detach(),
        rtol=1e-5,
        atol=1e-6,
    )


def main():
    dist.init_process_group(backend="gloo")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size == 2, "Launch this test with two ranks"
        test_sharded_linear(rank, world_size)
        if rank == 0:
            print(
                "PASS: ShardedLinear forward, backward, and AdamW updates "
                "match dense global-batch training",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
