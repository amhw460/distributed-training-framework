# Test global gradient clipping for replicated and sharded parameters.
# Run:
# torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \\
# --master-addr=127.0.0.1 --master-port=29503 -m tests.test_hybrid_clipping

import torch
import torch.distributed as dist
from torch import nn

from framework.fsdp import ShardedLinear
from framework.gradient_clipping import clip_hybrid_grad_norm_


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.replicated = nn.Linear(2, 2, bias=False)
        self.sharded = ShardedLinear(nn.Linear(2, 3, bias=False))


def test_clipping():
    rank = dist.get_rank()
    model = HybridModel()

    # The replicated gradient is [3, 4, 0, 0] on both ranks: norm squared 25.
    model.replicated.weight.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    # Each rank owns 3/6 sharded elements, with different gradient values.
    model.sharded.weightshard.grad = (
        torch.tensor([1.0, 2.0, 2.0]) if rank == 0
        else torch.tensor([2.0, 0.0, 0.0])
    )
    # Sharded squared norm = 9 + 4 = 13. Total = 38.
    original_norm = 38.0 ** 0.5
    result = clip_hybrid_grad_norm_(model, max_norm=2.0)
    assert abs(result - original_norm) < 1e-6
    scale = 2.0 / (original_norm + 1e-6)
    torch.testing.assert_close(model.replicated.weight.grad, torch.tensor([[3.0, 4.0], [0.0, 0.0]]) * scale)
    expected_shard = (
        torch.tensor([1.0, 2.0, 2.0]) if rank == 0
        else torch.tensor([2.0, 0.0, 0.0])
    )
    torch.testing.assert_close(model.sharded.weightshard.grad, expected_shard * scale)

    # With a large threshold the gradients must remain unchanged.
    result = clip_hybrid_grad_norm_(model, max_norm=100.0)
    assert abs(result - 2.0) < 1e-5


def main():
    dist.init_process_group("gloo")
    try:
        assert dist.get_world_size() == 2
        test_clipping()
        if dist.get_rank() == 0:
            print("PASS: hybrid replicated/sharded global gradient clipping", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
