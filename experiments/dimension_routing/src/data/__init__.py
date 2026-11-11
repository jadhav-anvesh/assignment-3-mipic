"""Data module for V2 routing."""

from routing_v2.src.data.dataset import (
    CANONICAL_TASKS,
    TASK_TO_ID,
    ID_TO_TASK,
    MultiTaskTextDataset,
    create_collate_fn,
)
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.data.validation import (
    EXPECTED_SOURCES,
    PAIR_SYMMETRIC_TASKS,
    PAIR_DIRECTED_TASKS,
    validate_training_csv,
    validate_eval_csv,
    validate_probe_class_balance,
    check_cross_split_leakage,
    check_train_corpus_overlap,
    map_scitail_label,
)

__all__ = [
    "CANONICAL_TASKS",
    "TASK_TO_ID",
    "ID_TO_TASK",
    "MultiTaskTextDataset",
    "create_collate_fn",
    "BalancedTaskBatchSampler",
    "EXPECTED_SOURCES",
    "PAIR_SYMMETRIC_TASKS",
    "PAIR_DIRECTED_TASKS",
    "validate_training_csv",
    "validate_eval_csv",
    "validate_probe_class_balance",
    "check_cross_split_leakage",
    "check_train_corpus_overlap",
    "map_scitail_label",
]
