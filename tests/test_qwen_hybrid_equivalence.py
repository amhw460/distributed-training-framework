# Compare full Qwen3-MoE training with DDP and hybrid DDP/FSDP.
# Run from the repository root (CPU/Gloo, no external dataset required):
# torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \\
# --master-addr=127.0.0.1 --master-port=29505 \\
# -m tests.test_qwen_hybrid_equivalence
# Checks forward outputs, accumulated and synchronized gradients, global gradient
# clipping, and updated model parameters over two optimizer steps.

import copy
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from framework.distributed import synchronize_gradients, synchronize_model
from framework.fsdp import gather_params, shard_model
from framework.gradient_clipping import clip_hybrid_grad_norm_
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs


def assert_model_weights_match(full, hybrid, rank, world_size):
    full_params = dict(full.named_parameters())
    hybrid_params = dict(hybrid.named_parameters())

    for name, full_param in full_params.items():
        if name == "outputprojection.weight":
            continue
        torch.testing.assert_close(hybrid_params[name].detach(), full_param.detach(), rtol=1e-4, atol=1e-6, msg=f"Replicated parameter mismatch: {name}")

    shard = hybrid.outputprojection
    reconstructed = gather_params(shard.weightshard.detach(), shard.shape, shard.numel)
    torch.testing.assert_close(reconstructed, full.outputprojection.weight.detach(), rtol=1e-4, atol=1e-6, msg="Reconstructed output projection mismatch")


def assert_gradients_match(full, hybrid, rank, world_size):
    full_params = dict(full.named_parameters())
    hybrid_params = dict(hybrid.named_parameters())

    for name, full_param in full_params.items():
        if name == "outputprojection.weight":
            continue

        hybrid_grad = hybrid_params[name].grad
        full_grad = full_param.grad
        assert (hybrid_grad is None) == (full_grad is None), (
            f"Different gradient presence for {name}"
        )
        if full_grad is not None:
            torch.testing.assert_close(hybrid_grad, full_grad, rtol=1e-4, atol=1e-6, msg=f"Replicated gradient mismatch: {name}")

    full_grad = full.outputprojection.weight.grad.flatten()
    local_shard = hybrid.outputprojection.weightshard.grad
    shard_size = math.ceil(full_grad.numel() / world_size)
    padded_grad = F.pad(full_grad, (0, shard_size * world_size - full_grad.numel()))
    expected_shard = padded_grad[rank * shard_size:(rank + 1) * shard_size]
    torch.testing.assert_close(local_shard, expected_shard, rtol=1e-4, atol=1e-6, msg="Sharded output projection gradient mismatch")


def test_qwen_training_equivalence(rank, world_size):
    # Small real Qwen3-MoE architecture, including attention and routed experts.
    args = Qwen3ModelArgs(vocab_size=37, max_seq_len=6, dim=8, moe_inter_dim=12, numlayers=1, numheads=2, numkvheads=1, head_dim=4, experts=3, k=2)
    torch.manual_seed(1234 + rank)
    full = Qwen3Model(args)
    synchronize_model(full)

    # Deepcopy gives both variants precisely the same initial weights.
    hybrid = copy.deepcopy(full)
    shard_model(hybrid)
    assert_model_weights_match(full, hybrid, rank, world_size)

    opt_full = torch.optim.AdamW(full.parameters(), lr=0.005, weight_decay=0.01)
    opt_hybrid = torch.optim.AdamW(hybrid.parameters(), lr=0.005, weight_decay=0.01)

    for step in range(2):
        opt_full.zero_grad()
        opt_hybrid.zero_grad()

        # Match the Trainer's two-microbatch gradient accumulation behavior.
        for micro in range(2):
            generator = torch.Generator().manual_seed(6000 + rank * 100 + step * 10 + micro)
            tokens = torch.randint(0, args.vocab_size, (2, 6), generator=generator)
            inputs, labels = tokens[:, :-1], tokens[:, 1:]

            full_logits = full(inputs)
            hybrid_logits = hybrid(inputs)
            torch.testing.assert_close(hybrid_logits, full_logits, rtol=1e-5, atol=1e-6, msg=f"Logit mismatch: step={step}, microbatch={micro}")

            full_loss = F.cross_entropy(
                full_logits.reshape(-1, args.vocab_size), labels.reshape(-1)
            ) / 2
            hybrid_loss = F.cross_entropy(
                hybrid_logits.reshape(-1, args.vocab_size), labels.reshape(-1)
            ) / 2
            full_loss.backward()
            hybrid_loss.backward()

        # The DDP reference all-reduces every weight; hybrid skips the shard.
        synchronize_gradients(full)
        synchronize_gradients(hybrid)
        assert_gradients_match(full, hybrid, rank, world_size)

        # Force clipping to also verify the global norm calculation.
        full_norm = clip_hybrid_grad_norm_(full, max_norm=0.1)
        hybrid_norm = clip_hybrid_grad_norm_(hybrid, max_norm=0.1)
        torch.testing.assert_close(torch.tensor(hybrid_norm), torch.tensor(full_norm), rtol=1e-4, atol=1e-6, msg=f"Global grad-norm mismatch at step={step}")
        assert_gradients_match(full, hybrid, rank, world_size)

        opt_full.step()
        opt_hybrid.step()
        assert_model_weights_match(full, hybrid, rank, world_size)


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size == 2, "Run with exactly two ranks"
        test_qwen_training_equivalence(rank, world_size)
        if rank == 0:
            print("PASS: Qwen3-MoE DDP and hybrid FSDP agree on logits, " "gradients, clipping, and weights after two updates", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
