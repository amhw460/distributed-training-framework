"""Smoke test GPipe global clipping and per-stage checkpoint round-trip.

Run: torchrun --standalone --nproc-per-node=2 -m tests.test_pp_training
"""
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from framework.gradient_clipping import clip_pp_grad_norm_
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.pipelineparallel import PipelineStage
from framework.pp_checkpoint import save_pp_checkpoint, load_pp_checkpoint
from framework.training.config import TrainingConfig
from framework.training.lr_scheduler import build_lr_scheduler


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank, total_nodes = dist.get_rank(), dist.get_world_size()
        args = Qwen3ModelArgs(vocab_size=29, max_seq_len=5, dim=8, moe_inter_dim=12,
                              numlayers=total_nodes + 1, numheads=4, numkvheads=2,
                              head_dim=4, experts=4, k=2)
        torch.manual_seed(300)
        stage = PipelineStage(Qwen3Model(args), rank, total_nodes)
        optimizer = torch.optim.AdamW(stage.parameters(), lr=0.001)
        config = TrainingConfig(steps=4, localbatchsize=3, globalbatchsize=3,
                                seqlen=4, learningrate=0.001, maxnorm=0.1,
                                seed=300, warmupsteps=1, decayratio=0.5,
                                decaytype="cosine", min_lr=0.1)
        scheduler = build_lr_scheduler(optimizer, config)

        # Assign a different artificial gradient to each rank to verify
        # that the global norm combines *disjoint* stage contributions.
        parameters = list(stage.parameters())
        for param in parameters:
            param.grad = torch.full_like(param, 0.01 * (rank + 1))
        local_sq = sum(p.grad.double().square().sum() for p in parameters)
        expected_sq = local_sq.clone()
        dist.all_reduce(expected_sq)
        expected_norm = expected_sq.sqrt()

        norm = clip_pp_grad_norm_(stage, 0.1)
        torch.testing.assert_close(torch.tensor(norm), expected_norm,
                                   rtol=1e-5, atol=1e-7)
        coeff = min(1.0, 0.1 / (expected_norm.item() + 1e-6))
        for param in parameters:
            torch.testing.assert_close(param.grad,
                                       torch.full_like(param, 0.01 * (rank + 1) * coeff),
                                       rtol=1e-5, atol=1e-7)
        optimizer.step()
        scheduler.step()

        # Separate local temporary directories per process avoid shared-fs
        # assumptions; each checkpoint path contains a rank-local file.
        with tempfile.TemporaryDirectory() as folder:
            saved = save_pp_checkpoint(folder, 1, stage, optimizer, scheduler,
                                       rank, total_nodes, args)
            expected_weights = {k: v.clone() for k, v in stage.state_dict().items()}
            expected_optimizer = optimizer.state_dict()
            expected_lr = scheduler.get_last_lr()

            for param in stage.parameters():
                param.data.add_(5)
            step = load_pp_checkpoint(saved, stage, optimizer, scheduler,
                                      rank, total_nodes, args)
            assert step == 1
            for name, actual in stage.state_dict().items():
                torch.testing.assert_close(actual, expected_weights[name])
            assert scheduler.get_last_lr() == expected_lr
            assert len(optimizer.state_dict()["state"]) == len(expected_optimizer["state"])

        dist.barrier()
        if rank == 0:
            print(f"PASS: PP global norm clipping and checkpoint round-trip, world={total_nodes}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
