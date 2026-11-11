"""Balanced multi-task batch sampler.

Ensures that every generated batch contains exactly `m` examples from each of the
configured tasks: Total Batch Size = num_tasks * m.
"""

from typing import Any, Dict, Iterator, List, Sequence
import numpy as np
from torch.utils.data import Sampler


class BalancedTaskBatchSampler(Sampler[List[int]]):
    """Yields batches containing an equal number of samples from each task.

    Args:
        source_ids: Sequence of task IDs for each sample in the dataset.
        examples_per_task: Number of examples (m) per task per batch.
        seed: Random seed for reproducible shuffling.
        num_tasks: Total number of expected tasks. Defaults to max(source_ids) + 1.
    """

    def __init__(
        self,
        source_ids: Sequence[int],
        examples_per_task: int = 4,
        seed: int = 42,
        num_tasks: int | None = None,
    ):
        self.source_ids = list(source_ids)
        self.examples_per_task = examples_per_task
        self.seed = seed
        self.epoch = 0

        # Group indices by task
        self.task_indices: Dict[int, List[int]] = {}
        for idx, sid in enumerate(self.source_ids):
            self.task_indices.setdefault(sid, []).append(idx)

        all_tasks = set(self.task_indices.keys())
        expected_tasks = set(range(num_tasks)) if num_tasks is not None else set(range(len(all_tasks)))

        missing_tasks = expected_tasks - all_tasks
        if missing_tasks:
            raise ValueError(
                f"Missing tasks in dataset: {sorted(missing_tasks)}. "
                "Cannot build balanced batches containing all tasks."
            )

        self.tasks = sorted(all_tasks)
        self.num_tasks = len(self.tasks)
        self.batch_size = self.num_tasks * self.examples_per_task

        # Check that each task has at least examples_per_task samples
        self.task_counts = {t: len(self.task_indices[t]) for t in self.tasks}
        for t, count in self.task_counts.items():
            if count < self.examples_per_task:
                raise ValueError(
                    f"Task {t} has only {count} samples, which is fewer than "
                    f"examples_per_task={self.examples_per_task}."
                )

        # Steps per epoch determined by the minimum task size
        min_task_count = min(self.task_counts.values())
        self.num_batches = min_task_count // self.examples_per_task

    def set_epoch(self, epoch: int) -> None:
        """Update epoch to generate different reproducible shuffles across epochs."""
        self.epoch = epoch

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics on samples used and dropped per epoch."""
        used_per_task = self.num_batches * self.examples_per_task
        return {
            "num_tasks": self.num_tasks,
            "examples_per_task_per_batch": self.examples_per_task,
            "total_batch_size": self.batch_size,
            "num_batches_per_epoch": self.num_batches,
            "total_samples_per_epoch": self.num_batches * self.batch_size,
            "task_counts": self.task_counts,
            "used_per_task": used_per_task,
            "unused_per_task": {t: self.task_counts[t] - used_per_task for t in self.tasks},
        }

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Deterministic RNG derived from base seed and current epoch
        rng = np.random.default_rng(self.seed + self.epoch * 10007)

        # Shuffle each task's pool independently
        shuffled_pools = {}
        for t in self.tasks:
            indices = np.array(self.task_indices[t], copy=True)
            rng.shuffle(indices)
            shuffled_pools[t] = indices

        m = self.examples_per_task
        for b in range(self.num_batches):
            batch = []
            for t in self.tasks:
                task_samples = shuffled_pools[t][b * m : (b + 1) * m]
                batch.extend(task_samples.tolist())

            # Shuffle the combined batch so tasks are uniformly interspersed
            rng.shuffle(batch)
            yield batch
