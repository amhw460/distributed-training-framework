"""Train Qwen3 with whatever parallelism dimensions the mesh specifies."""
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from framework.expertparallel import ExpertParallel
from framework.fsdp import (
    ShardedEmbedding, ShardedExperts, ShardedLinear,
    ShardedRMSNorm, ShardedTensorParallelLinear,
)
from framework.pipelineparallel import forward_microbatch, backward_microbatch
from framework.tensorparallel import (
    ColumnwiseLinearParallelism, RowwiseLinearParallelism,
    synchronize_tp_head_norm_gradients,
)


FSDP_TYPES = (
    ShardedEmbedding, ShardedExperts, ShardedLinear,
    ShardedRMSNorm, ShardedTensorParallelLinear,
)
TP_TYPES = (ColumnwiseLinearParallelism, RowwiseLinearParallelism)


def _expert_parameter_ids(model):
    return {id(p)
            for module in model.modules() if isinstance(module, ExpertParallel)
            for p in module.experts.parameters()}


def _tp_parameter_ids(model):
    ids = {id(module.weight) for module in model.modules() if isinstance(module, TP_TYPES)}
    ids.update(id(module.weightshard) for module in model.modules()
               if isinstance(module, ShardedTensorParallelLinear))
    return ids


def _fsdp_parameter_ids(model):
    return {id(module.weightshard) for module in model.modules()
            if isinstance(module, FSDP_TYPES)}


def synchronize_hybrid_gradients(model, mesh):
    """Synchronize only replicated dimensions; never mix unrelated shards."""
    params = list(model.parameters())
    expert_ids = _expert_parameter_ids(model)

    # rmsnorm weights are shared across tp ranks, so we need to sync their gradients
    if mesh.size("tp") > 1:
        layers = model.layers if hasattr(model, "layers") else model.transformerlayers
        class LocalModel:
            transformerlayers = layers
        synchronize_tp_head_norm_gradients(LocalModel(), group=mesh.group("tp"))

    # cp ranks each see different positions of the same sequence
    if mesh.size("cp") > 1:
        for p in params:
            # some experts might not get any tokens, but all ranks still need to sync
            present = torch.tensor(int(p.grad is not None), device=p.device)
            dist.all_reduce(present, group=mesh.group("cp"))
            if present.item() == 0:
                continue
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(grad, group=mesh.group("cp"))
            grad.div_(mesh.size("cp"))
            if p.grad is None:
                p.grad = grad

    # expert weights belong to their ep rank; shared weights need all-reduce
    if mesh.size("ep") > 1:
        for p in params:
            if id(p) in expert_ids:
                if p.grad is not None:
                    p.grad.div_(mesh.size("ep"))
                continue
            present = torch.tensor(int(p.grad is not None), device=p.device)
            dist.all_reduce(present, group=mesh.group("ep"))
            if present.item() == 0:
                continue
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(grad, group=mesh.group("ep"))
            grad.div_(mesh.size("ep"))
            if p.grad is None:
                p.grad = grad

    # fsdp handles its own gradient sync; dp ranks still need to average
    if mesh.size("dp") > 1:
        for p in params:
            present = torch.tensor(int(p.grad is not None), device=p.device)
            dist.all_reduce(present, group=mesh.group("dp"))
            if present.item() == 0:
                continue
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(grad, group=mesh.group("dp"))
            grad.div_(mesh.size("dp"))
            if p.grad is None:
                p.grad = grad


def clip_hybrid_gradients(model, mesh, max_norm):
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    expert_ids = _expert_parameter_ids(model)
    tp_ids = _tp_parameter_ids(model)
    fsdp_ids = _fsdp_parameter_ids(model)
    params = list(model.parameters())
    if not params:
        raise ValueError("model has no parameters")
    sq = torch.zeros((), dtype=torch.float64, device=params[0].device)

    # only count each logical parameter once, even if it is replicated
    for p in params:
        if p.grad is None:
            continue
        factor = 1.0 / (mesh.size("dp") * mesh.size("cp"))
        if id(p) not in expert_ids:
            factor /= mesh.size("ep")
        if id(p) not in tp_ids:
            factor /= mesh.size("tp")
        if id(p) not in fsdp_ids:
            factor /= mesh.size("fsdp")
        sq += p.grad.detach().double().square().sum() * factor

    dist.all_reduce(sq, op=dist.ReduceOp.SUM)
    norm = sq.sqrt().item()
    coefficient = min(1.0, max_norm / (norm + 1e-6))
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.grad.mul_(coefficient)
    return norm


def data_replica_coordinate(mesh):
    return (mesh.rank("dp"), mesh.rank("fsdp"), mesh.rank("ep"))


def replica_index(mesh):
    dp, fsdp, ep = data_replica_coordinate(mesh)
    return (dp * mesh.size("fsdp") + fsdp) * mesh.size("ep") + ep


def data_replica_count(mesh):
    return mesh.size("dp") * mesh.size("fsdp") * mesh.size("ep")


def create_batch_group(mesh):
    """Share a batch across pp, cp and tp ranks of the same data replica."""
    from itertools import product
    my_group = None
    my_leader = None
    for dp, fsdp, ep in product(range(mesh.size("dp")), range(mesh.size("fsdp")), range(mesh.size("ep"))):
        members = []
        for pp, cp, tp in product(range(mesh.size("pp")), range(mesh.size("cp")), range(mesh.size("tp"))):
            members.append(mesh._global_rank(dict(dp=dp, fsdp=fsdp, pp=pp, cp=cp, ep=ep, tp=tp)))
        group = dist.new_group(ranks=members)
        if mesh.global_rank in members:
            my_group = group
            my_leader = members[0]
    return my_group, my_leader


def broadcast_replica_batch(batch, mesh, batch_group, leader, shape, device):
    """Broadcast the replica's tokens to the other model-parallel ranks."""
    if mesh.global_rank != leader:
        batch = torch.empty(shape, device=device, dtype=torch.long)
    elif tuple(batch.shape) != tuple(shape):
        raise ValueError(f"Expected batch shape {shape}, received {tuple(batch.shape)}")
    dist.broadcast(batch, src=leader, group=batch_group)
    return batch


class HybridTrainer:
    def __init__(self, model, mesh, optimizer, max_norm, microbatches=1, scheduler=None, run_config=None):
        self.model = model
        self.mesh = mesh
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_config = dict(run_config or {})
        self.tokens_seen = 0
        self.last_metrics = {}
        self.max_norm = max_norm
        self.microbatches = microbatches
        self.step = 0
        if microbatches < 1:
            raise ValueError("microbatches must be >= 1")

    def _forward_backward(self, tokens, loss_scale=1.0):
        mesh = self.mesh
        x, y = tokens[:, :-1], tokens[:, 1:]
        if mesh.size("cp") > 1:
            x = x.tensor_split(mesh.size("cp"), dim=1)[mesh.rank("cp")].contiguous()
            y = y.tensor_split(mesh.size("cp"), dim=1)[mesh.rank("cp")].contiguous()
        if x.shape[1] == 0 or x.shape[1] != y.shape[1]:
            raise ValueError("Invalid context-parallel sequence partition")
        if x.shape[0] % self.microbatches:
            raise ValueError("Batch size must be divisible by pipeline microbatches")

        if mesh.size("pp") == 1:
            logits = self.model(x)
            loss = F.cross_entropy(logits.flatten(0, -2), y.flatten()) * loss_scale
            loss.backward()
            return loss.detach()

        # GPipe: all forward microbatches followed by reverse backward order.
        micro_x = x.tensor_split(self.microbatches, dim=0)
        micro_y = y.tensor_split(self.microbatches, dim=0)
        saved = []
        is_first = mesh.rank("pp") == 0
        is_last = mesh.rank("pp") == mesh.size("pp") - 1
        dtype = next(self.model.parameters()).dtype
        for xb, yb in zip(micro_x, micro_y):
            incoming, output = forward_microbatch(
                self.model, xb if is_first else None, mesh.rank("pp"), mesh.size("pp"),
                xb.size(0), xb.size(1), self.model.layers[0].model_args.dim,
                xb.device, dtype, group=mesh.group("pp"),
            )
            loss = (
                F.cross_entropy(output.flatten(0, -2), yb.flatten(), reduction="sum")
                / y.numel() * loss_scale
                if is_last else None
            )
            saved.append((incoming, output, loss))
        for incoming, output, loss in reversed(saved):
            backward_microbatch(incoming, output, mesh.rank("pp"), mesh.size("pp"),
                                loss=loss, group=mesh.group("pp"))
        if is_last:
            return sum(loss.detach() for _, _, loss in saved)
        return torch.zeros((), device=x.device)

    def training_step(self, tokens):
        batches = tokens if isinstance(tokens, list) else [tokens]
        if not batches:
            raise ValueError("At least one local batch is required")
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.zero_grad(set_to_none=True)
        loss = sum(self._forward_backward(batch, 1.0 / len(batches)) for batch in batches)
        synchronize_hybrid_gradients(self.model, self.mesh)
        norm = clip_hybrid_gradients(self.model, self.mesh, self.max_norm)
        self.optimizer.step()
        self.step += 1
        if self.scheduler is not None:
            self.scheduler.step()
        batch_tokens = sum(batch.shape[0] * (batch.shape[1] - 1) for batch in batches)
        global_tokens = batch_tokens * data_replica_count(self.mesh)
        self.tokens_seen += global_tokens
        # average the loss across replicas, including ranks on other pipeline stages
        value = loss.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= self.mesh.world_size // self.mesh.size("pp")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        duration = torch.tensor(elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        elapsed = duration.item()
        self.last_metrics = {
            "step": self.step,
            "loss": value.item(),
            "grad_norm": norm,
            "lr": lr,
            "step_time_s": elapsed,
            "tokens_per_sec": global_tokens / elapsed,
            "tokens_seen": self.tokens_seen,
            "step_tokens": global_tokens,
        }
        return value.item(), norm

    def save(self, folder, model_args):
        """Save this rank's weights and training state so we can resume later."""
        directory = Path(folder) / f"step-{self.step}"
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": self.step,
            "rank": self.mesh.global_rank,
            "mesh": self.mesh.sizes,
            "model_args": vars(model_args).copy(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "run_config": self.run_config,
            "tokens_seen": self.tokens_seen,
            "rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng"] = torch.cuda.get_rng_state()
        final = directory / f"rank-{self.mesh.global_rank}.pt"
        temp = directory / f"rank-{self.mesh.global_rank}.tmp"
        torch.save(payload, temp)
        temp.replace(final)
        dist.barrier()
        return directory

    def load(self, directory, model_args):
        payload = torch.load(Path(directory) / f"rank-{self.mesh.global_rank}.pt",
                             map_location="cpu", weights_only=True)
        if payload["rank"] != self.mesh.global_rank or payload["mesh"] != self.mesh.sizes:
            raise ValueError("Checkpoint rank or mesh does not match")
        if payload["model_args"] != vars(model_args):
            raise ValueError("Checkpoint model configuration does not match")
        if payload.get("run_config", {}) != self.run_config:
            raise ValueError("Checkpoint training configuration does not match")
        if (payload.get("scheduler") is None) != (self.scheduler is None):
            raise ValueError("Checkpoint scheduler configuration does not match")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        torch.set_rng_state(payload["rng"])
        if "cuda_rng" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state(payload["cuda_rng"])
        self.step = int(payload["step"])
        dist.barrier()
        return self.step
