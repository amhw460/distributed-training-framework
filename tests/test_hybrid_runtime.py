# Check optimizer steps, metrics and checkpoint resume against a full run.
import argparse
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.parallel_mesh import ParallelMesh
from framework.model_composer import compose_model
from framework.hybrid_train import HybridTrainer, replica_index, data_replica_count
from framework.training.runtime import build_optimizer, build_hybrid_scheduler


def make_trainer(mesh, args):
    torch.manual_seed(501)
    model = compose_model(Qwen3Model(args), mesh, args.max_seq_len).model
    optimizer = build_optimizer(model.parameters(), name="adamw", lr=0.002, weight_decay=0.02)
    schedule = build_hybrid_scheduler(optimizer, name="cosine", steps=3, warmup_steps=1, decay_ratio=0.5, min_lr_factor=0.2)
    return HybridTrainer(model, mesh, optimizer, max_norm=0.5, microbatches=2,
                         scheduler=schedule, run_config={"test": "scheduled"})


def batch(mesh, step):
    generator = torch.Generator().manual_seed(900 + 100 * replica_index(mesh) + step)
    return torch.randint(0, 37, (2, 9), generator=generator)


def run():
    parser = argparse.ArgumentParser()
    for name in ParallelMesh.DIMENSIONS:
        parser.add_argument("--" + name, type=int, default=1)
    flags = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        mesh = ParallelMesh(**vars(flags))
        args = Qwen3ModelArgs(vocab_size=37, max_seq_len=8, dim=8, moe_inter_dim=12, numlayers=4, numheads=4, numkvheads=2, head_dim=4, experts=4, k=2)
        uninterrupted = make_trainer(mesh, args)
        losses = []
        for step in range(3):
            loss, _ = uninterrupted.training_step(batch(mesh, step))
            losses.append(loss)
        expected = {k: v.detach().clone() for k, v in uninterrupted.model.state_dict().items()}
        assert uninterrupted.step == 3
        assert uninterrupted.tokens_seen == 3 * 2 * 8 * data_replica_count(mesh)
        assert uninterrupted.last_metrics["step_tokens"] == 2 * 8 * data_replica_count(mesh)
        assert uninterrupted.last_metrics["lr"] > 0
        assert uninterrupted.last_metrics["step_time_s"] > 0
        assert uninterrupted.last_metrics["tokens_per_sec"] > 0

        # use the same checkpoint folder on all ranks without object broadcasts
        folder = Path(tempfile.gettempdir()) / f"hybrid-resume-{os.environ['MASTER_PORT']}"
        interrupted = make_trainer(mesh, args)
        interrupted.training_step(batch(mesh, 0))
        checkpoint = interrupted.save(folder, args)
        restored = make_trainer(mesh, args)
        restored.load(checkpoint, args)
        assert restored.step == 1
        assert restored.tokens_seen == 2 * 8 * data_replica_count(mesh)
        for step in range(1, 3):
            restored.training_step(batch(mesh, step))
        for name, tensor in restored.model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], rtol=1e-5, atol=1e-6, msg=name)
        assert restored.step == uninterrupted.step
        assert restored.tokens_seen == uninterrupted.tokens_seen
        assert restored.optimizer.param_groups[0]["lr"] == uninterrupted.optimizer.param_groups[0]["lr"]
        assert restored.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
        if mesh.global_rank == 0:
            print("PASS: scheduled training, metrics and checkpoint-resume equivalence", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run()
