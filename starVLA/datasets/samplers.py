"""Batch samplers for LCLGP training (T-W2.3.3).

Provides :class:`TaskGroupedSampler`: every emitted batch contains
``n_tasks`` distinct tasks, each contributing ``n_demos`` samples. This
guarantees the LCLGP K-mode multimodal head sees task diversity within a
mini-batch (D1 in the design doc) and is what `pav_libero` / Bridge-v2
mixtures route through during training.

Why a custom sampler at all?
----------------------------
Stock :class:`torch.utils.data.RandomSampler` shuffles globally; with K=4
multimodal modes, it is easy for two adjacent batches to share zero tasks,
which starves the contrastive InfoNCE term and makes mode-balancing noisy.
Grouping by task on each batch fixes both problems.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class TaskGroupedSampler(Sampler[List[int]]):
    """Yield batches of ``n_tasks * n_demos`` indices grouped by task.

    Args:
        task_ids: per-sample task id of length ``len(dataset)``. Hashable
            objects (typically int or str). The sampler does not look at the
            dataset itself.
        n_tasks: distinct tasks per batch.
        n_demos: samples drawn per task per batch.
        num_batches: total batches per epoch. Defaults to roughly one pass
            over the dataset (``len(dataset) // (n_tasks * n_demos)``).
        shuffle_demos_with_replacement: when a task has fewer than
            ``n_demos`` samples, fall back to sampling with replacement.
            Default ``True``.
        drop_remainder: ignored — every emitted batch is full by construction.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        task_ids: Sequence,
        *,
        n_tasks: int,
        n_demos: int,
        num_batches: int | None = None,
        shuffle_demos_with_replacement: bool = True,
        seed: int = 0,
    ):
        if n_tasks < 1 or n_demos < 1:
            raise ValueError("n_tasks and n_demos must both be >= 1")

        self._task_ids = list(task_ids)
        self.n_tasks = n_tasks
        self.n_demos = n_demos
        self.shuffle_demos_with_replacement = shuffle_demos_with_replacement
        self._seed = seed
        self._epoch = 0

        # Bucket sample indices by task.
        self._task_to_indices: dict = defaultdict(list)
        for idx, t in enumerate(self._task_ids):
            self._task_to_indices[t].append(idx)
        self._unique_tasks = list(self._task_to_indices.keys())

        if len(self._unique_tasks) < n_tasks:
            raise ValueError(
                f"only {len(self._unique_tasks)} distinct tasks available, "
                f"cannot satisfy n_tasks={n_tasks}"
            )

        if num_batches is None:
            num_batches = max(1, len(self._task_ids) // (n_tasks * n_demos))
        self.num_batches = num_batches

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Match :class:`torch.utils.data.distributed.DistributedSampler` API."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self._seed + self._epoch)
        unique_tasks = np.asarray(self._unique_tasks, dtype=object)
        for _ in range(self.num_batches):
            chosen_tasks = rng.choice(unique_tasks, size=self.n_tasks, replace=False)
            batch: List[int] = []
            for t in chosen_tasks:
                pool = self._task_to_indices[t.item() if hasattr(t, "item") else t]
                if len(pool) >= self.n_demos:
                    picks = rng.choice(pool, size=self.n_demos, replace=False)
                elif self.shuffle_demos_with_replacement:
                    picks = rng.choice(pool, size=self.n_demos, replace=True)
                else:
                    raise RuntimeError(
                        f"task {t} has only {len(pool)} samples (< n_demos={self.n_demos}); "
                        f"set shuffle_demos_with_replacement=True or filter the task out"
                    )
                batch.extend(int(i) for i in picks)
            yield batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.n_tasks * self.n_demos

    @property
    def num_unique_tasks(self) -> int:
        return len(self._unique_tasks)

    def task_summary(self) -> List[tuple]:
        """Return ``[(task_id, count), ...]`` sorted by count descending."""
        return sorted(
            ((t, len(idxs)) for t, idxs in self._task_to_indices.items()),
            key=lambda x: -x[1],
        )

    @staticmethod
    def estimate_num_batches(num_samples: int, n_tasks: int, n_demos: int) -> int:
        """Convenience for callers that prefer to pass num_batches explicitly."""
        return max(1, math.floor(num_samples / (n_tasks * n_demos)))
