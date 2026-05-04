"""DDP-aware variant of :class:`TaskGroupedSampler` for LCLGP training.

Mirrors :class:`torch.utils.data.distributed.DistributedSampler` semantics:
each rank emits a disjoint slice of the dataset's tasks every epoch, and
``set_epoch(e)`` reseeds reproducibly across ranks.

With 8 ranks and the default LCLGP setup (32 tasks × 8 demos = global batch
of 256), each rank yields 4 tasks × 8 demos = 32 samples per batch, and the
4 tasks held by any one rank are disjoint from the 4 tasks held by every
other rank in that batch — the InfoNCE term then sees 32 rank-local negatives
plus, after ``accelerator.gather``, all 256 cross-task negatives.

Why not just wrap the existing TaskGroupedSampler in DistributedSampler?
``DistributedSampler`` shards *indices*, but TaskGroupedSampler emits
*batches* (lists of indices). Sharding batches loses the task-grouping
invariant. We instead shard *tasks* (the bigger granular unit) and rebuild
batches per-rank.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class DistributedTaskGroupedSampler(Sampler[List[int]]):
    """Yield ``n_tasks_per_rank * n_demos`` indices per batch on each rank.

    Args:
        task_ids: per-sample task id of length ``len(dataset)``.
        n_tasks_per_rank: distinct tasks each rank gets per batch.
        n_demos: samples drawn per task per batch.
        rank, world_size: standard distributed identifiers.
        num_batches: batches per rank per epoch. Defaults to roughly one pass
            over the rank's share of the dataset.
        shuffle_demos_with_replacement: see :class:`TaskGroupedSampler`.
        seed: base RNG seed; shifted by ``epoch`` on ``set_epoch``.
    """

    def __init__(
        self,
        task_ids: Sequence,
        *,
        n_tasks_per_rank: int,
        n_demos: int,
        rank: int,
        world_size: int,
        num_batches: int | None = None,
        shuffle_demos_with_replacement: bool = True,
        seed: int = 0,
    ):
        if n_tasks_per_rank < 1 or n_demos < 1:
            raise ValueError("n_tasks_per_rank and n_demos must both be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")

        self._task_ids = list(task_ids)
        self.n_tasks_per_rank = n_tasks_per_rank
        self.n_demos = n_demos
        self.rank = rank
        self.world_size = world_size
        self.shuffle_demos_with_replacement = shuffle_demos_with_replacement
        self._seed = seed
        self._epoch = 0

        self._task_to_indices: dict = defaultdict(list)
        for idx, t in enumerate(self._task_ids):
            self._task_to_indices[t].append(idx)
        self._unique_tasks = list(self._task_to_indices.keys())

        n_tasks_global = n_tasks_per_rank * world_size
        if len(self._unique_tasks) < n_tasks_global:
            raise ValueError(
                f"only {len(self._unique_tasks)} distinct tasks available, "
                f"need >= n_tasks_per_rank * world_size = {n_tasks_global}"
            )

        if num_batches is None:
            per_rank = len(self._task_ids) // world_size
            num_batches = max(1, per_rank // (n_tasks_per_rank * n_demos))
        self.num_batches = num_batches

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # All ranks share the same RNG → shuffle the global task list identically;
        # then slice rank-disjoint chunks. Each rank's per-demo picks use a different
        # RNG so demos vary per rank.
        global_rng = np.random.default_rng(self._seed + self._epoch)
        local_rng = np.random.default_rng(self._seed + self._epoch * 1000 + self.rank + 1)
        unique_tasks = np.asarray(self._unique_tasks, dtype=object)
        n_tasks_global = self.n_tasks_per_rank * self.world_size

        for _ in range(self.num_batches):
            chosen_tasks = global_rng.choice(unique_tasks, size=n_tasks_global, replace=False)
            start = self.rank * self.n_tasks_per_rank
            my_tasks = chosen_tasks[start : start + self.n_tasks_per_rank]

            batch: List[int] = []
            for t in my_tasks:
                pool = self._task_to_indices[t.item() if hasattr(t, "item") else t]
                if len(pool) >= self.n_demos:
                    picks = local_rng.choice(pool, size=self.n_demos, replace=False)
                elif self.shuffle_demos_with_replacement:
                    picks = local_rng.choice(pool, size=self.n_demos, replace=True)
                else:
                    raise RuntimeError(
                        f"task {t} has only {len(pool)} samples (< n_demos={self.n_demos})"
                    )
                batch.extend(int(i) for i in picks)
            yield batch

    @property
    def batch_size(self) -> int:
        return self.n_tasks_per_rank * self.n_demos

    @staticmethod
    def estimate_num_batches(
        num_samples: int, world_size: int, n_tasks_per_rank: int, n_demos: int
    ) -> int:
        per_rank = num_samples // world_size
        return max(1, math.floor(per_rank / (n_tasks_per_rank * n_demos)))
