"""Linear probing models for sentence and pair classification tasks."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from routing_v2.src.evaluation.metrics import compute_macro_f1, compute_spearman


def select_best_c(
    x_train: np.ndarray,
    y_train: np.ndarray,
    candidate_c: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    max_iter: int = 1000,
    seed: int = 42,
    return_score: bool = False,
    standardize: bool = False,
) -> Any:
    """Select the best regularization C via stratified cross-validation on probe train data.
    
    Strict Isolation: Any preprocessing (e.g. StandardScaler) is fitted strictly inside each
    CV training fold. No validation fold or test labels ever influence C selection.
    Parallelized across candidate C values using multi-threading for fast evaluation.
    """
    if len(np.unique(y_train)) < 2 or len(x_train) < 10:
        return (1.0, 0.0) if return_score else 1.0

    n_splits = min(3, int(np.min(np.bincount(y_train))))
    if n_splits < 2:
        return (1.0, 0.0) if return_score else 1.0

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(x_train, y_train))

    def _eval_single_c(c: float) -> Tuple[float, float]:
        scores = []
        for tr_idx, val_idx in folds:
            x_tr_fold = x_train[tr_idx]
            x_val_fold = x_train[val_idx]
            if standardize:
                scaler = StandardScaler()
                x_tr_fold = scaler.fit_transform(x_tr_fold)
                x_val_fold = scaler.transform(x_val_fold)
            clf = LogisticRegression(C=c, max_iter=max_iter, random_state=seed)
            clf.fit(x_tr_fold, y_train[tr_idx])
            preds = clf.predict(x_val_fold)
            scores.append(compute_macro_f1(y_train[val_idx], preds))
        return c, float(np.mean(scores))

    # Evaluate candidates in parallel across available CPU cores
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_eval_single_c)(c) for c in candidate_c
    )

    best_c = candidate_c[0]
    best_score = -1.0
    for c, avg_score in results:
        if avg_score > best_score:
            best_score = avg_score
            best_c = c

    if return_score:
        return best_c, best_score
    return best_c


def fit_and_eval_single_probe(
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    test_embs: np.ndarray,
    test_labels: np.ndarray,
    tune_c: bool = True,
    max_iter: int = 1000,
    seed: int = 42,
    return_val: bool = False,
    standardize: bool = False,
    return_c: bool = False,
) -> Any:
    """Train logistic regression on train split and return macro-F1 on test split (and optional CV val score / C)."""
    if tune_c:
        best_c, val_score = select_best_c(
            x_train=train_embs,
            y_train=train_labels,
            max_iter=max_iter,
            seed=seed,
            return_score=True,
            standardize=standardize,
        )
    else:
        best_c = 1.0
        val_score = 0.0

    if standardize:
        scaler = StandardScaler()
        train_feat = scaler.fit_transform(train_embs)
        test_feat = scaler.transform(test_embs)
    else:
        train_feat = train_embs
        test_feat = test_embs

    clf = LogisticRegression(C=best_c, max_iter=max_iter, random_state=seed)
    clf.fit(train_feat, train_labels)
    preds = clf.predict(test_feat)
    test_score = float(compute_macro_f1(test_labels, preds))

    if return_c and return_val:
        return test_score, float(val_score), float(best_c)
    elif return_c:
        return test_score, float(best_c)
    elif return_val:
        return test_score, float(val_score)
    return test_score


def compute_pair_features(
    u: np.ndarray,
    v: np.ndarray,
    feature_mode: str = "diff",
) -> np.ndarray:
    """Compute sentence pair features.
    
    Modes:
        diff: |u - v| (standard probing feature)
        diff_prod: [|u - v|, u * v]
    """
    diff = np.abs(u - v)
    if feature_mode == "diff_prod":
        prod = u * v
        return np.concatenate([diff, prod], axis=-1)
    return diff


def fit_and_eval_pair_probe(
    train_embs1: np.ndarray,
    train_embs2: np.ndarray,
    train_labels: np.ndarray,
    test_embs1: np.ndarray,
    test_embs2: np.ndarray,
    test_labels: np.ndarray,
    feature_mode: str = "diff",
    tune_c: bool = True,
    max_iter: int = 1000,
    seed: int = 42,
    return_val: bool = False,
    return_c: bool = False,
) -> Any:
    """Evaluate sentence pair classification using difference features."""
    train_feat = compute_pair_features(train_embs1, train_embs2, feature_mode=feature_mode)
    test_feat = compute_pair_features(test_embs1, test_embs2, feature_mode=feature_mode)

    return fit_and_eval_single_probe(
        train_embs=train_feat,
        train_labels=train_labels,
        test_embs=test_feat,
        test_labels=test_labels,
        tune_c=tune_c,
        max_iter=max_iter,
        seed=seed,
        return_val=return_val,
        return_c=return_c,
    )


def eval_sts_regression(
    test_embs1: np.ndarray,
    test_embs2: np.ndarray,
    gold_scores: np.ndarray,
) -> float:
    """Evaluate STS similarity using cosine similarity and Spearman correlation."""
    u_norm = np.linalg.norm(test_embs1, axis=1) + 1e-8
    v_norm = np.linalg.norm(test_embs2, axis=1) + 1e-8
    cos_sim = np.sum(test_embs1 * test_embs2, axis=1) / (u_norm * v_norm)
    return compute_spearman(cos_sim, gold_scores)

