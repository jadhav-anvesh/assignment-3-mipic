"""Full cross-evaluation engine for all tasks and dimension configurations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from routing_v2.src.evaluation.embeddings import extract_embeddings
from routing_v2.src.evaluation.probes import (
    fit_and_eval_single_probe,
    fit_and_eval_pair_probe,
    eval_sts_regression,
)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize vectors along the last dimension to unit norm (L2)."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def evaluate_task_on_features(
    task_name: str,
    task_cfg: Dict[str, Any],
    train_cache: Dict[str, Any],
    test_cache: Dict[str, Any],
    slice_fn: Any,
    tune_c: bool = False,
    seed: int = 42,
    return_val: bool = False,
    return_c: bool = False,
) -> Any:
    """Evaluate one task for a given embedding slice function."""
    task_type = task_cfg["type"]

    if task_type == "single":
        tr_embs = slice_fn(train_cache["embs"])
        te_embs = slice_fn(test_cache["embs"])
        return fit_and_eval_single_probe(
            train_embs=tr_embs,
            train_labels=train_cache["labels"],
            test_embs=te_embs,
            test_labels=test_cache["labels"],
            tune_c=tune_c,
            seed=seed,
            return_val=return_val,
            return_c=return_c,
        )

    elif task_type == "pair":
        tr_1 = slice_fn(train_cache["embs1"])
        tr_2 = slice_fn(train_cache["embs2"])
        te_1 = slice_fn(test_cache["embs1"])
        te_2 = slice_fn(test_cache["embs2"])
        return fit_and_eval_pair_probe(
            train_embs1=tr_1,
            train_embs2=tr_2,
            train_labels=train_cache["labels"],
            test_embs1=te_1,
            test_embs2=te_2,
            test_labels=test_cache["labels"],
            tune_c=tune_c,
            seed=seed,
            return_val=return_val,
            return_c=return_c,
        )

    elif task_type == "pair_regression":
        te_1 = slice_fn(test_cache["embs1"])
        te_2 = slice_fn(test_cache["embs2"])
        test_score = eval_sts_regression(
            test_embs1=te_1,
            test_embs2=te_2,
            gold_scores=test_cache["labels"],
        )
        if return_val:
            if "embs1" in train_cache and "embs2" in train_cache:
                tr_1 = slice_fn(train_cache["embs1"])
                tr_2 = slice_fn(train_cache["embs2"])
                val_score = eval_sts_regression(
                    test_embs1=tr_1,
                    test_embs2=tr_2,
                    gold_scores=train_cache["labels"],
                )
            else:
                val_score = test_score
            if return_c:
                return test_score, val_score, 1.0
            return test_score, val_score
        if return_c:
            return test_score, 1.0
        return test_score

    else:
        raise ValueError(f"Unknown task type '{task_type}' for task {task_name}")


def evaluate_all_tasks(
    model: torch.nn.Module,
    tokenizer: object,
    tasks_config: Dict[str, Dict[str, Any]],
    dimension_mode: str = "disjoint",
    blocks: Optional[List[Tuple[int, int]]] = None,
    nested_dims: Optional[Tuple[int, ...]] = (16, 32, 64, 128, 256, 512, 768),
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    tune_c: bool = False,
    seed: int = 42,
    return_val_scores: bool = False,
    normalize_prefixes: bool = True,
    return_c: bool = False,
) -> Any:
    """Evaluate full embedding model across all tasks and block/nested slices.
    
    Extracts embeddings ONCE per task, then slices in memory for fast evaluation.
    For nested mode, each prefix z_{:, :d} is independently L2-normalized after slicing.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    results_matrix: Dict[str, Dict[str, float]] = {}
    val_results_matrix: Dict[str, Dict[str, float]] = {}
    c_matrix: Dict[str, Dict[str, float]] = {}

    for task_name, cfg in tasks_config.items():
        task_type = cfg["type"]
        train_cache: Dict[str, Any] = {}
        test_cache: Dict[str, Any] = {}

        # Load test set (and train set if classification probe or pair regression)
        test_df = pd.read_csv(cfg["test"])

        if task_type == "single":
            train_df = pd.read_csv(cfg["train"])
            train_cache["embs"] = extract_embeddings(
                model, train_df["text"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
            )
            train_cache["labels"] = train_df["label"].values

            test_cache["embs"] = extract_embeddings(
                model, test_df["text"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
            )
            test_cache["labels"] = test_df["label"].values

        elif task_type in ("pair", "pair_regression"):
            if "train" in cfg:
                train_df = pd.read_csv(cfg["train"])
                train_cache["embs1"] = extract_embeddings(
                    model, train_df["sentence1"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
                )
                train_cache["embs2"] = extract_embeddings(
                    model, train_df["sentence2"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
                )
                train_cache["labels"] = train_df["label"].values.astype(float) if task_type == "pair_regression" else train_df["label"].values

            test_cache["embs1"] = extract_embeddings(
                model, test_df["sentence1"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
            )
            test_cache["embs2"] = extract_embeddings(
                model, test_df["sentence2"].astype(str).tolist(), tokenizer, batch_size=batch_size, device=device
            )
            test_cache["labels"] = test_df["label"].values.astype(float)

        task_scores: Dict[str, float] = {}
        task_val_scores: Dict[str, float] = {}
        task_c_scores: Dict[str, float] = {}

        if dimension_mode == "disjoint":
            assert blocks is not None, "blocks must be provided for disjoint dimension_mode"
            for b_idx, (start, end) in enumerate(blocks):
                res = evaluate_task_on_features(
                    task_name=task_name,
                    task_cfg=cfg,
                    train_cache=train_cache,
                    test_cache=test_cache,
                    slice_fn=lambda x, s=start, e=end: x[:, s:e],
                    tune_c=tune_c,
                    seed=seed,
                    return_val=return_val_scores,
                    return_c=return_c,
                )
                if return_c and return_val_scores:
                    task_scores[str(b_idx)] = float(res[0])
                    task_val_scores[str(b_idx)] = float(res[1])
                    task_c_scores[str(b_idx)] = float(res[2])
                elif return_val_scores:
                    task_scores[str(b_idx)] = float(res[0])
                    task_val_scores[str(b_idx)] = float(res[1])
                elif return_c:
                    task_scores[str(b_idx)] = float(res[0])
                    task_c_scores[str(b_idx)] = float(res[1])
                else:
                    task_scores[str(b_idx)] = float(res)

        elif dimension_mode == "nested":
            assert nested_dims is not None, "nested_dims must be provided for nested dimension_mode"
            for dim in nested_dims:
                slice_fn = (lambda x, d=dim: l2_normalize_np(x[:, :d])) if normalize_prefixes else (lambda x, d=dim: x[:, :d])
                res = evaluate_task_on_features(
                    task_name=task_name,
                    task_cfg=cfg,
                    train_cache=train_cache,
                    test_cache=test_cache,
                    slice_fn=slice_fn,
                    tune_c=tune_c,
                    seed=seed,
                    return_val=return_val_scores,
                    return_c=return_c,
                )
                if return_c and return_val_scores:
                    task_scores[str(dim)] = float(res[0])
                    task_val_scores[str(dim)] = float(res[1])
                    task_c_scores[str(dim)] = float(res[2])
                elif return_val_scores:
                    task_scores[str(dim)] = float(res[0])
                    task_val_scores[str(dim)] = float(res[1])
                elif return_c:
                    task_scores[str(dim)] = float(res[0])
                    task_c_scores[str(dim)] = float(res[1])
                else:
                    task_scores[str(dim)] = float(res)
        else:
            raise ValueError(f"Unknown dimension_mode '{dimension_mode}'")

        results_matrix[task_name] = task_scores
        if return_val_scores:
            val_results_matrix[task_name] = task_val_scores
        if return_c:
            c_matrix[task_name] = task_c_scores

    if return_c and return_val_scores:
        return results_matrix, val_results_matrix, c_matrix
    elif return_val_scores:
        return results_matrix, val_results_matrix
    elif return_c:
        return results_matrix, c_matrix
    return results_matrix

