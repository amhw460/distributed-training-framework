# Full Qwen3-MoE FSDP versus replicated DDP correctness test.
# Run from repository root:
# torchrun --nnodes=1 --nproc-per-node=2 --rdzv-backend=static \\
# --master-addr=127.0.0.1 --master-port=29506 -m tests.test_full_fsdp
# No FineWeb data or GPU needed.

import copy
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from framework.distributed import synchronize_gradients, synchronize_model
from framework.fsdp import (
    ShardedExperts, ShardedLinear, ShardedEmbedding, ShardedRMSNorm,
    gather_params, shard_model,
)
from framework.gradient_clipping import clip_hybrid_grad_norm_
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs


SHARDED_TYPES = (ShardedLinear, ShardedEmbedding, ShardedRMSNorm, ShardedExperts)


# Rebuild all parameters in the same order/names as the dense Qwen model.
def materialize_sharded(sharded):
    result = {}
    for module_name, module in sharded.named_modules():
        if not isinstance(module, SHARDED_TYPES):
            continue
        recovered = gather_params(module.weightshard.detach(), module.shape, module.numel)
        if isinstance(module, ShardedExperts):
            matrices = torch.split(recovered.reshape(-1), module.weight_numels)
            for i in range(module.num_experts):
                for j, proj in enumerate(("up_proj", "gate", "down_proj")):
                    idx = 3 * i + j
                    result[f"{module_name}.{i}.{proj}.weight"] = matrices[idx].reshape(module.weight_shapes[idx])
        else:
            result[f"{module_name}.weight"] = recovered
    return result


def assert_sharded_weights_equal(dense, sharded):
    expected = dict(dense.named_parameters())
    actual = materialize_sharded(sharded)
    assert set(actual) == set(expected), (
        f"Not all dense weights were sharded: "
        f"missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}"
    )
    for name, parameter in expected.items():
        torch.testing.assert_close(actual[name], parameter.detach(), rtol=2e-4, atol=2e-6, msg=f"Weight mismatch: {name}")


def assert_sharded_grads_equal(dense, sharded, rank, world_size):
    dense_modules = dict(dense.named_modules())
    for module_name, module in sharded.named_modules():
        if not isinstance(module, SHARDED_TYPES):
            continue
        if isinstance(module, ShardedExperts):
            dense_experts = dense_modules[module_name]
            full_grad = torch.cat([
                getattr(expert, proj).weight.grad.flatten()
                if getattr(expert, proj).weight.grad is not None
                else torch.zeros_like(getattr(expert, proj).weight).flatten()
                for expert in dense_experts
                for proj in ("up_proj", "gate", "down_proj")
            ])
        else:
            full_grad = dense_modules[module_name].weight.grad.flatten()

        local_grad = module.weightshard.grad
        assert local_grad is not None, f"Missing shard gradient: {module_name}"
        shard_size = math.ceil(full_grad.numel() / world_size)
        padded = F.pad(full_grad, (0, shard_size * world_size - full_grad.numel()))
        expected = padded[rank * shard_size:(rank + 1) * shard_size]
        torch.testing.assert_close(local_grad, expected, rtol=2e-4, atol=2e-6, msg=f"Shard gradient mismatch: {module_name}")


def test_full_fsdp(rank, world_size):
    args = Qwen3ModelArgs(vocab_size=37, max_seq_len=6, dim=8, moe_inter_dim=12, numlayers=2, numheads=2, numkvheads=1, head_dim=4, experts=4, k=2)

    torch.manual_seed(12345 + rank)
    dense = Qwen3Model(args)
    synchronize_model(dense)
    sharded = copy.deepcopy(dense)
    shard_model(sharded)

    # The sharded model should not contain *any* dense registered parameters.
    assert all(isinstance(m, SHARDED_TYPES) for m in (
        module for module in sharded.modules()
        if any(p is not None for p in module.parameters(recurse=False))
    ))
    assert_sharded_weights_equal(dense, sharded)

    opt_dense = torch.optim.AdamW(dense.parameters(), lr=0.005, weight_decay=0.01)
    opt_sharded = torch.optim.AdamW(sharded.parameters(), lr=0.005, weight_decay=0.01)

    for step in range(2):
        opt_dense.zero_grad()
        opt_sharded.zero_grad()
        for micro in range(2):
            generator = torch.Generator().manual_seed(8000 + 1000 * rank + 10 * step + micro)
            tokens = torch.randint(0, args.vocab_size, (2, 6), generator=generator)
            inputs, labels = tokens[:, :-1], tokens[:, 1:]
            dense_logits = dense(inputs)
            sharded_logits = sharded(inputs)
            torch.testing.assert_close(sharded_logits, dense_logits, rtol=2e-4, atol=2e-6, msg=f"Logits differ: step={step}, microbatch={micro}")
            dense_loss = F.cross_entropy(
                dense_logits.reshape(-1, args.vocab_size), labels.reshape(-1)
            ) / 2
            sharded_loss = F.cross_entropy(
                sharded_logits.reshape(-1, args.vocab_size), labels.reshape(-1)
            ) / 2
            dense_loss.backward()
            sharded_loss.backward()

        synchronize_gradients(dense)
        synchronize_gradients(sharded)
        assert_sharded_grads_equal(dense, sharded, rank, world_size)

        reference_norm = clip_hybrid_grad_norm_(dense, max_norm=0.1)
        actual_norm = clip_hybrid_grad_norm_(sharded, max_norm=0.1)
        torch.testing.assert_close(torch.tensor(actual_norm), torch.tensor(reference_norm), rtol=2e-4, atol=2e-6, msg=f"Global clipping norm mismatch step={step}")
        assert_sharded_grads_equal(dense, sharded, rank, world_size)
        opt_dense.step()
        opt_sharded.step()
        assert_sharded_weights_equal(dense, sharded)


def main():
    torch.set_num_threads(1)
    dist.init_process_group(backend="gloo")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        assert world_size == 2, "Use exactly two ranks"
        test_full_fsdp(rank, world_size)
        if rank == 0:
            print("PASS: full-model FSDP matches DDP on all weights, " "forward outputs, backward gradients, clipping, and AdamW updates", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
