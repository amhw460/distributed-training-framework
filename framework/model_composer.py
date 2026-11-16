"""Compose existing Qwen3 parallelism primitives in a fixed order.

The mesh selects which algorithms run; combinations are not enumerated.
"""
from dataclasses import dataclass
from framework.hybrid import HybridPlan
from framework.pipelineparallel import PipelineStage
from framework.expertparallel import ExpertParallel
from framework.tensorparallel import parallelize_attn, parallelize_moe
from framework.contextparallel import parallelize_cp
from framework.fsdp import shard_local_modules


@dataclass
class ComposedModel:
    model: object
    plan: HybridPlan


def compose_model(model, mesh, sequence_length):
    plan = HybridPlan(mesh, model.model_args, sequence_length)
    # Pipeline stage partitioning must occur before local tensor partitioning.
    if mesh.size("pp") > 1:
        local = PipelineStage(model, mesh.rank("pp"), mesh.size("pp"))
        layers = local.layers
    else:
        local = model
        layers = local.transformerlayers

    if mesh.size("ep") > 1:
        for layer in layers:
            layer.moe = ExpertParallel(
                layer.moe, mesh.rank("ep"), mesh.size("ep"), group=mesh.group("ep")
            )

    if mesh.size("tp") > 1:
        for layer in layers:
            parallelize_attn(layer.attention, group=mesh.group("tp"))
            parallelize_moe(layer.moe, group=mesh.group("tp"))

    if mesh.size("cp") > 1:
        if mesh.size("pp") > 1:
            local.cp_rank = mesh.rank("cp")
            for layer in layers:
                layer.attention.cp_mode = True
                layer.attention.cp_group = mesh.group("cp")
        else:
            parallelize_cp(local, group=mesh.group("cp"))

    if mesh.size("fsdp") > 1:
        local = shard_local_modules(local, group=mesh.group("fsdp"))

    return ComposedModel(local, plan)
