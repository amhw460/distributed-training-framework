"""Apply the selected parallelism strategies to the Qwen3 model."""

from dataclasses import dataclass

from framework.contextparallel import parallelize_cp
from framework.expertparallel import ExpertParallel
from framework.fsdp import shard_local_modules
from framework.hybrid import HybridPlan
from framework.pipelineparallel import PipelineStage
from framework.tensorparallel import parallelize_attn, parallelize_moe


@dataclass
class ComposedModel:
    model: object
    plan: HybridPlan


def compose_model(model, mesh, sequence_length):
    plan = HybridPlan(mesh, model.model_args, sequence_length)

    # split the layers into pipeline stages before sharding anything else
    if mesh.size("pp") > 1:
        local = PipelineStage(model, mesh.rank("pp"), mesh.size("pp"))
        layers = local.layers
    else:
        local = model
        layers = local.transformerlayers

    # each ep rank owns a different set of experts
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

    # fsdp shards whatever weights are left on this rank
    if mesh.size("fsdp") > 1:
        local = shard_local_modules(local, group=mesh.group("fsdp"))

    return ComposedModel(local, plan)
