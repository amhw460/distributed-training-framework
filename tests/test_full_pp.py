"""Compare GPipe Qwen3-MoE training against full-model training on CPU/Gloo.

Run from the repository root:
    torchrun --standalone --nproc-per-node=1 -m tests.test_full_pp
    torchrun --standalone --nproc-per-node=2 -m tests.test_full_pp
    torchrun --standalone --nproc-per-node=3 -m tests.test_full_pp

Tests stage ownership, loss, local parameter gradients, and AdamW updates.
"""
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.pipelineparallel import PipelineStage, gpipe_step


def configurations(total_nodes):
    return [
        (Qwen3ModelArgs(vocab_size=37, max_seq_len=7, dim=8, moe_inter_dim=12,
            numlayers=total_nodes, numheads=4, numkvheads=2, head_dim=4,
            experts=4, k=2), 5, 2),
        (Qwen3ModelArgs(vocab_size=43, max_seq_len=6, dim=12, moe_inter_dim=10,
            numlayers=2 * total_nodes + 1, numheads=6, numkvheads=2, head_dim=4,
            experts=5, k=3), 7, 3),
    ]


def dense_name(local_name, stage, total_nodes):
    if not local_name.startswith("layers."):
        return local_name
    q, r = divmod(stage.num_original_layers, total_nodes)
    start = stage.rank * q + min(stage.rank, r)
    return "transformerlayers." + str(start + int(local_name.split(".")[1])) + "." + ".".join(local_name.split(".")[2:])


def compare_stage(dense, stage, total_nodes, compare_gradients=True):
    full_params = dict(dense.named_parameters())
    local_params = dict(stage.named_parameters())
    assert local_params, "Pipeline stage should have trainable parameters"
    for local_name, local_param in local_params.items():
        name = dense_name(local_name, stage, total_nodes)
        ref = full_params[name]
        torch.testing.assert_close(local_param.detach(), ref.detach(), rtol=5e-4, atol=3e-6,
                                   msg=f"rank={stage.rank} weight {name}")
        if compare_gradients:
            assert (local_param.grad is None) == (ref.grad is None), f"rank={stage.rank} gradient presence {name}"
            if ref.grad is not None:
                torch.testing.assert_close(local_param.grad, ref.grad, rtol=5e-4, atol=3e-6,
                                           msg=f"rank={stage.rank} gradient {name}")


def run_case(config, batch_size, num_microbatches, rank, total_nodes, case_number):
    torch.manual_seed(1000 + case_number)
    dense = Qwen3Model(config)
    # Partition a separate identical model; retain dense only as a test oracle.
    stage = PipelineStage(copy.deepcopy(dense), rank, total_nodes)
    stage.num_original_layers = config.numlayers

    q, r = divmod(config.numlayers, total_nodes)
    assert len(stage.layers) == q + (rank < r)
    assert (stage.embeddings is not None) == (rank == 0)
    assert (stage.outputprojection is not None) == (rank == total_nodes - 1)
    assert (stage.finalRMSnorm is not None) == (rank == total_nodes - 1)
    compare_stage(dense, stage, total_nodes, compare_gradients=False)

    dense_optimizer = torch.optim.AdamW(dense.parameters(), lr=0.003, weight_decay=0.01)
    stage_optimizer = torch.optim.AdamW(stage.parameters(), lr=0.003, weight_decay=0.01)

    for step in range(2):
        generator = torch.Generator().manual_seed(3300 + 100 * case_number + step)
        tokens = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len), generator=generator)

        dense_optimizer.zero_grad(set_to_none=True)
        dense_logits = dense(tokens[:, :-1])
        dense_loss = F.cross_entropy(dense_logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
        dense_loss.backward()

        # GPipe independently accumulates gradients across (possibly uneven) microbatches.
        actual_loss = gpipe_step(stage, tokens if rank == 0 else None, stage_optimizer,
                                 num_microbatches, rank, total_nodes, config.dim, torch.device("cpu"))
        if rank == total_nodes - 1:
            torch.testing.assert_close(torch.tensor(actual_loss), dense_loss.detach(),
                                       rtol=5e-4, atol=3e-6, msg=f"case={case_number} step={step} loss")

        # The GPipe optimizer stepped, but its gradient tensors remain available.
        # Check gradients first; only the dense optimizer has not stepped yet.
        full_params = dict(dense.named_parameters())
        for local_name, local_param in stage.named_parameters():
            name = dense_name(local_name, stage, total_nodes)
            ref = full_params[name]
            assert (local_param.grad is None) == (ref.grad is None), f"gradient presence {name}"
            if ref.grad is not None:
                torch.testing.assert_close(local_param.grad, ref.grad, rtol=5e-4, atol=3e-6,
                                           msg=f"case={case_number} step={step} rank={rank} gradient {name}")

        dense_optimizer.step()
        compare_stage(dense, stage, total_nodes)

    dist.barrier()
    if rank == 0:
        print(f"PASS: GPipe world={total_nodes}, layers={config.numlayers}, "
              f"batch={batch_size}, microbatches={num_microbatches}, two AdamW steps", flush=True)


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank, total_nodes = dist.get_rank(), dist.get_world_size()
        for i, (config, batch_size, num_microbatches) in enumerate(configurations(total_nodes)):
            run_case(config, batch_size, num_microbatches, rank, total_nodes, i)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
