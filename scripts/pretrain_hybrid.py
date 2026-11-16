# Train Qwen3 MoE with any of the supported mesh parallelism dimensions.

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from framework.hybrid_train import (
    HybridTrainer, broadcast_replica_batch, create_batch_group,
    data_replica_count, replica_index,
)
from framework.model_composer import compose_model
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs
from framework.parallel_mesh import ParallelMesh
from framework.training.runtime import build_hybrid_scheduler, build_optimizer


def options():
    p = argparse.ArgumentParser()
    for axis in ParallelMesh.DIMENSIONS:
        p.add_argument("--" + axis, type=int, default=1)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--local-batch-size", type=int, default=2)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=127)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--moe-inter-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=8)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=("adamw", "adam", "sgd"), default="adamw")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--scheduler", choices=("constant", "linear", "sqrt", "cosine"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--decay-ratio", type=float)
    p.add_argument("--min-lr-factor", type=float, default=0.0)
    p.add_argument("--metrics-file", type=Path)
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--resume", type=Path)
    return p.parse_args()


def main():
    cfg = options()
    if cfg.steps < 1 or cfg.local_batch_size < 1 or cfg.seq_len < 1:
        raise ValueError("steps, local batch size, and sequence length must be positive")
    if cfg.microbatches < 1 or cfg.microbatches > cfg.local_batch_size or cfg.local_batch_size % cfg.microbatches:
        raise ValueError("microbatches must divide local batch size")
    if cfg.num_layers < 1 or cfg.dim < 1 or cfg.moe_inter_dim < 1:
        raise ValueError("model dimensions must be positive")
    if cfg.num_heads < 1 or cfg.num_kv_heads < 1 or cfg.num_heads % cfg.num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if cfg.head_dim < 2 or cfg.head_dim % 2:
        raise ValueError("head dimension must be positive and even")
    if cfg.experts < 1 or not 1 <= cfg.top_k <= cfg.experts:
        raise ValueError("Require 1 <= top-k <= experts")
    if cfg.lr <= 0 or cfg.max_norm <= 0 or cfg.save_every < 0:
        raise ValueError("invalid learning rate, max norm, or checkpoint interval")
    if cfg.warmup_steps < 0 or cfg.warmup_steps > cfg.steps:
        raise ValueError("warmup steps must fit within training steps")
    if cfg.save_every and not cfg.checkpoint_dir:
        raise ValueError("--save-every requires --checkpoint-dir")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            torch.set_num_threads(1)

        mesh = ParallelMesh(**{axis: getattr(cfg, axis) for axis in ParallelMesh.DIMENSIONS})
        group, leader = create_batch_group(mesh)
        num_replicas = data_replica_count(mesh)
        nominal_global = cfg.local_batch_size * num_replicas
        global_batch = cfg.global_batch_size or nominal_global
        if global_batch < nominal_global or global_batch % nominal_global:
            raise ValueError("global batch must be a multiple of local batch * DP * FSDP * EP")
        accumulation = global_batch // nominal_global
        if cfg.seq_len % mesh.size("cp"):
            raise ValueError("CP must divide sequence length")

        torch.manual_seed(cfg.seed)
        if cfg.synthetic:
            if cfg.vocab_size < 2:
                raise ValueError("synthetic vocabulary must have at least 2 tokens")
            vocab_size = cfg.vocab_size
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            vocab_size = len(tokenizer)

        args = Qwen3ModelArgs(vocab_size=vocab_size, max_seq_len=cfg.seq_len, dim=cfg.dim, moe_inter_dim=cfg.moe_inter_dim, numlayers=cfg.num_layers, numheads=cfg.num_heads, numkvheads=cfg.num_kv_heads, head_dim=cfg.head_dim, experts=cfg.experts, k=cfg.top_k)
        dense = Qwen3Model(args).to(device)
        # start from the same weights on every rank before we shard the model
        composed = compose_model(dense, mesh, cfg.seq_len)
        model = composed.model
        del dense
        optimizer = build_optimizer(model.parameters(), name=cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps, momentum=cfg.momentum)
        scheduler = build_hybrid_scheduler(optimizer, name=cfg.scheduler, steps=cfg.steps, warmup_steps=cfg.warmup_steps, decay_ratio=cfg.decay_ratio, min_lr_factor=cfg.min_lr_factor)
        resume_config = {
            "optimizer": cfg.optimizer, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
            "beta1": cfg.beta1, "beta2": cfg.beta2, "eps": cfg.eps, "momentum": cfg.momentum,
            "scheduler": cfg.scheduler, "warmup_steps": cfg.warmup_steps,
            "decay_ratio": cfg.decay_ratio, "min_lr_factor": cfg.min_lr_factor,
            "global_batch_size": global_batch, "local_batch_size": cfg.local_batch_size,
            "microbatches": cfg.microbatches, "seed": cfg.seed,
        }
        if cfg.scheduler != "constant":
            resume_config["schedule_steps"] = cfg.steps
        trainer = HybridTrainer(model, mesh, optimizer, cfg.max_norm, cfg.microbatches,
                                scheduler=scheduler, run_config=resume_config)
        if cfg.resume:
            trainer.load(cfg.resume, args)
        if trainer.step > cfg.steps:
            raise ValueError("Resume checkpoint is beyond requested steps")

        dataset = None
        if not cfg.synthetic and mesh.global_rank == leader:
            # each dp/fsdp/ep replica loads its own data
            from framework.data.dataset import build_fineweb_dataset
            dataset = build_fineweb_dataset(tokenizer, cfg.seq_len, replica_index(mesh), num_replicas)
            dataset.samples_seen = trainer.step * accumulation * cfg.local_batch_size
            source = iter(DataLoader(dataset, batch_size=cfg.local_batch_size, drop_last=True, num_workers=0))

        if mesh.global_rank == 0:
            print(composed.plan.describe(), flush=True)

        for step in range(trainer.step, cfg.steps):
            step_started = time.perf_counter()
            batches = []
            for micro in range(accumulation):
                batch = None
                if mesh.global_rank == leader:
                    if cfg.synthetic:
                        seed = cfg.seed + 100000 * replica_index(mesh) + 1000 * step + micro
                        generator = torch.Generator().manual_seed(seed)
                        batch = torch.randint(0, vocab_size, (cfg.local_batch_size, cfg.seq_len + 1), generator=generator, dtype=torch.long, device="cpu").to(device)
                    else:
                        batch = next(source).to(device)
                batch = broadcast_replica_batch(batch, mesh, group, leader, (cfg.local_batch_size, cfg.seq_len + 1), device)
                batches.append(batch)

            loss, norm = trainer.training_step(batches)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # this includes loading and broadcasting data, not just the training step
            elapsed = torch.tensor(time.perf_counter() - step_started, dtype=torch.float64, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            trainer.last_metrics["end_to_end_tokens_per_sec"] = trainer.last_metrics["step_tokens"] / elapsed.item()
            if mesh.global_rank == 0:
                metrics = trainer.last_metrics
                print(f"step={metrics['step']} loss={metrics['loss']:.6f} " f"grad_norm={metrics['grad_norm']:.6f} lr={metrics['lr']:.8g} " f"tokens_seen={metrics['tokens_seen']} " f"step_time_s={metrics['step_time_s']:.4f} " f"tokens_per_sec={metrics['tokens_per_sec']:.2f} " f"end_to_end_tokens_per_sec={metrics['end_to_end_tokens_per_sec']:.2f}", flush=True)
                if cfg.metrics_file is not None:
                    cfg.metrics_file.parent.mkdir(parents=True, exist_ok=True)
                    with cfg.metrics_file.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metrics) + "\n")
            if cfg.checkpoint_dir and cfg.save_every and trainer.step % cfg.save_every == 0:
                trainer.save(cfg.checkpoint_dir, args)

        if mesh.global_rank == 0:
            print("PASS: hybrid training completed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
