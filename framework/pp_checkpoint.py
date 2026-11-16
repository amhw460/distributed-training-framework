# Per-rank pipeline checkpoints. Requires the same rank count and model partition on resume.
from pathlib import Path

import torch
import torch.distributed as dist


def save_pp_checkpoint(folder, step, stage, optimizer, scheduler, rank, total_nodes, model_args):
    directory = Path(folder) / f"step-{step}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "rank": rank,
        "total_nodes": total_nodes,
        "model_args": vars(model_args).copy(),
        "model": stage.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": torch.get_rng_state(),
    }
    final_path = directory / f"rank-{rank}.pt"
    temporary_path = directory / f"rank-{rank}.tmp"
    torch.save(payload, temporary_path)
    temporary_path.replace(final_path)
    if dist.is_initialized(): dist.barrier()
    return directory


def load_pp_checkpoint(directory, stage, optimizer, scheduler, rank, total_nodes, model_args):
    directory = Path(directory)
    state = torch.load(directory / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
    if state["rank"] != rank or state["total_nodes"] != total_nodes:
        raise ValueError("Pipeline rank/count mismatch in checkpoint")
    if state["model_args"] != vars(model_args):
        raise ValueError("Model configuration differs from checkpoint")

    stage.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    if dist.is_initialized(): dist.barrier()
    return int(state["step"])
