"""Set up ranks and communication groups for every parallelism dimension."""

from itertools import product
from math import prod

import torch.distributed as dist


class ParallelMesh:
    # the last dimension changes fastest when assigning global ranks
    DIMENSIONS = ("dp", "fsdp", "pp", "cp", "ep", "tp")

    def __init__(self, dp=1, fsdp=1, pp=1, cp=1, ep=1, tp=1):
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before creating a mesh")

        self.sizes = dict(zip(self.DIMENSIONS, (dp, fsdp, pp, cp, ep, tp)))
        if any(type(size) is not int or size < 1 for size in self.sizes.values()):
            raise ValueError("Every mesh dimension must be a positive integer")

        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        if prod(self.sizes.values()) != self.world_size:
            raise ValueError("Product of mesh dimensions must equal world size")

        # each rank gets one coordinate for each kind of parallelism
        remaining = self.global_rank
        self.coordinates = {}
        for dimension in reversed(self.DIMENSIONS):
            size = self.sizes[dimension]
            self.coordinates[dimension] = remaining % size
            remaining //= size

        self._groups = {}
        self._group_ranks = {}

        # every rank has to create these groups in the same order
        for dimension in self.DIMENSIONS:
            other_dimensions = [name for name in self.DIMENSIONS if name != dimension]
            for other_coordinates in product(
                *(range(self.sizes[name]) for name in other_dimensions)
            ):
                coordinates = dict(zip(other_dimensions, other_coordinates))
                ranks = []
                for local_rank in range(self.sizes[dimension]):
                    coordinates[dimension] = local_rank
                    ranks.append(self._global_rank(coordinates))

                group = dist.new_group(ranks=ranks)
                if self.global_rank in ranks:
                    self._groups[dimension] = group
                    self._group_ranks[dimension] = tuple(ranks)

    def _global_rank(self, coordinates):
        rank = 0
        for dimension in self.DIMENSIONS:
            rank = rank * self.sizes[dimension] + coordinates[dimension]
        return rank

    def group(self, dimension):
        return self._groups[dimension]

    def ranks(self, dimension):
        return self._group_ranks[dimension]

    def rank(self, dimension):
        return self.coordinates[dimension]

    def size(self, dimension):
        return self.sizes[dimension]
