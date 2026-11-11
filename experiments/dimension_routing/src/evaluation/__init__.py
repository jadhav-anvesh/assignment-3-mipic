"""Evaluation module for V2 routing."""

from routing_v2.src.evaluation.embeddings import extract_embeddings
from routing_v2.src.evaluation.metrics import (
    compute_macro_f1,
    compute_spearman,
    compute_summary_statistics,
)
from routing_v2.src.evaluation.probes import (
    fit_and_eval_single_probe,
    fit_and_eval_pair_probe,
    eval_sts_regression,
    compute_pair_features,
)
from routing_v2.src.evaluation.evaluate import evaluate_all_tasks

__all__ = [
    "extract_embeddings",
    "compute_macro_f1",
    "compute_spearman",
    "compute_summary_statistics",
    "fit_and_eval_single_probe",
    "fit_and_eval_pair_probe",
    "eval_sts_regression",
    "compute_pair_features",
    "evaluate_all_tasks",
]

