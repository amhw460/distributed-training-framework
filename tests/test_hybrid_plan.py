"""Check which ranks own the layers and parameter shards."""
import argparse

import torch
import torch.distributed as dist

from framework.hybrid import HybridPlan
from framework.parallel_mesh import ParallelMesh
from framework.models.model import Qwen3Model
from framework.models.model_args import Qwen3ModelArgs


def check_plan(mesh, args, plan, dense):
    # Every rank agrees on the same global rank-to-coordinate mapping.
    assert len(plan.active_strategies) == sum(mesh.size(a) > 1 for a in mesh.DIMENSIONS)
    for rank in range(mesh.world_size):
        place = plan.placement(rank)
        assert mesh._global_rank(place.coordinates) == rank
        assert place.embedding_owner == (place.coordinates["pp"] == 0)
        assert place.output_owner == (place.coordinates["pp"] == mesh.size("pp") - 1)

    for layer in range(args.numlayers):
        for pp in range(mesh.size("pp")):
            # each layer should belong to exactly one pipeline stage
            expected = [rank for rank in range(mesh.world_size)
                                    if layer in plan.placement(rank).layers]
            if pp == 0:
                assert len(expected) == mesh.world_size // mesh.size("pp")

    for e in range(args.experts):
        assert {plan.placement(rank).coordinates["ep"]
                        for rank in range(mesh.world_size)
                        if e in plan.placement(rank).experts} == {e // (args.experts // mesh.size("ep"))}

    for name, param in dense.named_parameters():
        owners = plan.parameter_owners(name, param.shape)
        assert owners, f"No rank owns {name}"
        for rank in range(mesh.world_size):
            placement = plan.parameter(name, param.shape, rank)
            assert placement.present == (rank in owners)
            if placement.tp_axis is None:
                assert placement.post_tp_shape == tuple(param.shape)
            else:
                axis = placement.tp_axis
                start, end = placement.tp_interval
                assert placement.post_tp_shape[axis] == end - start
                assert end > start, (name, rank)

        # tp shards should cover the full tensor without overlapping
        for rank in owners:
            place = plan.placement(rank)
            if place.coordinates["tp"] != 0: continue
            shards = [plan.parameter(name, param.shape, peer)
                                for peer in mesh_ranks_for_coordinate(plan, place.coordinates, "tp")]
            assert all(shard.present for shard in shards)
            if shards[0].tp_axis is not None:
                segments = sorted(shard.tp_interval for shard in shards)
                assert segments[0][0] == 0
                assert segments[-1][1] == param.shape[shards[0].tp_axis]
                assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))


def mesh_ranks_for_coordinate(plan, coordinates, varying_axis):
    return [rank for rank in range(plan.mesh.world_size)
                    if all(plan.placement(rank).coordinates[a] == coordinates[a]
                                  for a in plan.mesh.DIMENSIONS if a != varying_axis)]


def check_validation(mesh, args):
    def rejected(**overrides):
        import copy
        broken = copy.copy(args)
        for key, value in overrides.items(): setattr(broken, key, value)
        try: HybridPlan(mesh, broken, sequence_length=8)
        except ValueError: return
        raise AssertionError(f"Expected rejection for {overrides}")

    if mesh.size("tp") > 1:
        rejected(numkvheads=mesh.size("tp") + 1)
    if mesh.size("ep") > 1:
        rejected(experts=mesh.size("ep") + 1)
    if mesh.size("pp") > 1:
        rejected(numlayers=mesh.size("pp") - 1)
    try:
        HybridPlan(mesh, args, sequence_length=args.max_seq_len + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Sequence longer than model maximum must fail")


def main():
    parser = argparse.ArgumentParser()
    for dim in ParallelMesh.DIMENSIONS:
        parser.add_argument("--" + dim, type=int, default=1)
    options = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        mesh = ParallelMesh(**vars(options))
        args = Qwen3ModelArgs(vocab_size=37, max_seq_len=8, dim=8,
                                                      moe_inter_dim=12, numlayers=4, numheads=4,
                                                      numkvheads=4, head_dim=4, experts=4, k=2)
        dense = Qwen3Model(args)
        plan = HybridPlan(mesh, args, sequence_length=8)
        check_plan(mesh, args, plan, dense)
        check_validation(mesh, args)
        # print the plan from one rank so we can inspect it
        if dist.get_rank() == 0:
            print(plan.describe(), flush=True)
            print("PASS: hybrid placement and parameter coverage", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
