"""Check rank coordinates and collectives within each mesh group."""

import argparse
from itertools import product

import torch
import torch.distributed as dist

from framework.parallel_mesh import ParallelMesh


def check(mesh):
    rank = dist.get_rank()
    assert mesh.global_rank == rank
    assert mesh.world_size == dist.get_world_size()
    assert mesh._global_rank(mesh.coordinates) == rank

    for dimension in mesh.DIMENSIONS:
        ranks = mesh.ranks(dimension)
        assert len(ranks) == mesh.size(dimension)
        assert ranks[mesh.rank(dimension)] == rank
        assert dist.get_world_size(group=mesh.group(dimension)) == mesh.size(dimension)
        assert dist.get_rank(group=mesh.group(dimension)) == mesh.rank(dimension)

        # the all-reduce should only sum ranks that belong to this group
        value = torch.tensor(float(rank + 1))
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=mesh.group(dimension))
        assert value.item() == sum(member + 1 for member in ranks), dimension

        # other mesh coordinates should be the same within a group
        local_coords = mesh.coordinates
        for other_rank in ranks:
            remaining = other_rank
            coordinates = {}
            for axis in reversed(mesh.DIMENSIONS):
                coordinates[axis] = remaining % mesh.size(axis)
                remaining //= mesh.size(axis)
            assert all(coordinates[axis] == local_coords[axis]
                                  for axis in mesh.DIMENSIONS if axis != dimension)


def main():
    parser = argparse.ArgumentParser()
    for name in ParallelMesh.DIMENSIONS:
        parser.add_argument("--" + name, type=int, default=1)
    args = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        mesh = ParallelMesh(**vars(args))
        check(mesh)
        if dist.get_rank() == 0:
            print("PASS: mesh dimensions, coordinates, groups and collectives", vars(args), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
