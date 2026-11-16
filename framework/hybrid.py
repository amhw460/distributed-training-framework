"""Figure out which layers, experts and tensor slices each rank owns."""

from dataclasses import dataclass
import re


def _partition(length, parts, index):
    """Split a dimension into almost equally sized chunks."""
    quotient, remainder = divmod(length, parts)
    start = index * quotient + min(index, remainder)
    end = start + quotient + (index < remainder)
    return start, end


@dataclass(frozen=True)
class RankPlacement:
    global_rank: int
    coordinates: dict
    layers: tuple
    experts: tuple
    query_heads: tuple
    kv_heads: tuple
    sequence_tokens: tuple
    input_replica: tuple
    embedding_owner: bool
    output_owner: bool


@dataclass(frozen=True)
class ParameterPlacement:
    name: str
    global_shape: tuple
    post_tp_shape: tuple
    tp_axis: int | None
    tp_interval: tuple | None
    fsdp_requested: bool
    present: bool


class HybridPlan:
    """Keep track of model placement before actually sharding weights."""

    def __init__(self, mesh, model_args, sequence_length):
        self.mesh = mesh
        self.args = model_args
        self.sequence_length = sequence_length
        self.sizes = mesh.sizes.copy()
        self.active_strategies = tuple(
            axis for axis in mesh.DIMENSIONS if mesh.size(axis) > 1
        )
        if sequence_length < 1 or sequence_length > model_args.max_seq_len:
            raise ValueError("sequence_length must be within model max_seq_len")
        if model_args.numlayers < mesh.size("pp"):
            raise ValueError("PP stages cannot outnumber transformer layers")
        if model_args.experts % mesh.size("ep"):
            raise ValueError("EP degree must divide number of experts")
        if model_args.numheads % mesh.size("tp") or model_args.numkvheads % mesh.size("tp"):
            raise ValueError("TP degree must divide Q and KV head counts")
        if model_args.moe_inter_dim < mesh.size("tp"):
            raise ValueError("TP degree must not exceed expert intermediate dimension")
        if sequence_length % mesh.size("cp"):
            raise ValueError("Sequence length must be divisible by CP degree")

    def _coords(self, rank):
        if not 0 <= rank < self.mesh.world_size:
            raise ValueError("rank outside mesh")
        coords = {}
        for axis in reversed(self.mesh.DIMENSIONS):
            coords[axis] = rank % self.mesh.size(axis)
            rank //= self.mesh.size(axis)
        return coords

    def placement(self, rank=None):
        if rank is None:
            rank = self.mesh.global_rank
        coords = self._coords(rank)
        start, end = _partition(
            self.args.numlayers, self.mesh.size("pp"), coords["pp"]
        )
        expert_start, expert_end = _partition(
            self.args.experts, self.mesh.size("ep"), coords["ep"]
        )
        q = _partition(self.args.numheads, self.mesh.size("tp"), coords["tp"])
        kv = _partition(self.args.numkvheads, self.mesh.size("tp"), coords["tp"])
        seq = _partition(self.sequence_length, self.mesh.size("cp"), coords["cp"])
        return RankPlacement(
            global_rank=rank,
            coordinates=coords,
            layers=tuple(range(start, end)),
            experts=tuple(range(expert_start, expert_end)),
            query_heads=q,
            kv_heads=kv,
            sequence_tokens=seq,
            # tp, cp and pp share a batch; dp, fsdp and ep get different data
            input_replica=(coords["dp"], coords["fsdp"], coords["ep"]),
            embedding_owner=coords["pp"] == 0,
            output_owner=coords["pp"] == self.mesh.size("pp") - 1,
        )

    def parameter(self, name, shape, rank=None):
        """Find where a parameter lives before we transform the model."""
        location = self.placement(rank)
        shape = tuple(shape)
        present = True
        tp_axis = None

        if name == "embeddings.weight":
            present = location.embedding_owner
        elif name in ("finalRMSnorm.weight", "outputprojection.weight"):
            present = location.output_owner
        else:
            match = re.fullmatch(r"transformerlayers\.(\d+)\.(.+)", name)
            if not match:
                raise ValueError(f"Unknown Qwen3 parameter: {name}")
            layer_index, suffix = int(match.group(1)), match.group(2)
            present = layer_index in location.layers

            expert = re.fullmatch(
                r"moe\.experts\.(\d+)\.(up_proj|gate|down_proj)\.weight", suffix
            )
            if expert:
                present &= int(expert.group(1)) in location.experts
                tp_axis = 1 if expert.group(2) == "down_proj" else 0
            elif suffix in ("attention.Wq.weight", "attention.Wk.weight", "attention.Wv.weight"):
                tp_axis = 0
            elif suffix == "attention.output.weight":
                tp_axis = 1
            elif suffix not in (
                "firstnorm.weight", "secondnorm.weight",
                "attention.rmsNormQ.weight", "attention.rmsNormK.weight",
                "moe.router.router.weight"
            ):
                raise ValueError(f"Unknown Qwen3 parameter: {name}")

        interval = None
        post_tp_shape = list(shape)
        if tp_axis is not None:
            interval = _partition(
                shape[tp_axis], self.mesh.size("tp"), location.coordinates["tp"]
            )
            post_tp_shape[tp_axis] = interval[1] - interval[0]

        return ParameterPlacement(
            name=name,
            global_shape=shape,
            post_tp_shape=tuple(post_tp_shape),
            tp_axis=tp_axis,
            tp_interval=interval,
            fsdp_requested=self.mesh.size("fsdp") > 1,
            present=present,
        )

    def parameter_owners(self, name, shape):
        """Ranks that hold at least part of this weight."""
        return tuple(
            rank for rank in range(self.mesh.world_size)
            if self.parameter(name, shape, rank).present
        )

    def describe(self):
        lines = [
            "Hybrid plan (placement only, not applied to a model)",
            "Active strategies: " + (", ".join(self.active_strategies) or "none"),
            "Dimensions: " + ", ".join(
                f"{axis}={self.mesh.size(axis)}" for axis in self.mesh.DIMENSIONS
            ),
        ]
        for rank in range(self.mesh.world_size):
            place = self.placement(rank)
            layers = f"{place.layers[0]}..{place.layers[-1]}"
            experts = f"{place.experts[0]}..{place.experts[-1]}"
            lines.append(
                f"rank {rank}: pp_layers={layers}, ep_experts={experts}, "
                f"tp_heads={place.query_heads}, tp_kv_heads={place.kv_heads}, "
                f"cp_tokens={place.sequence_tokens}, input_replica={place.input_replica}, "
                f"embeddings={place.embedding_owner}, output_head={place.output_owner}"
            )
        return "\n".join(lines)
