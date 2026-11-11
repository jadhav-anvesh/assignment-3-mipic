"""Data validation module for V2 routing.

Ensures strict schema validation, label verification, absence of data leakage,
task-aware reversed-pair checks, probe class balance, and generation of a
complete dataset manifest before training starts.
"""

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import hashlib
import pandas as pd

EXPECTED_SOURCES: Set[str] = {
    "TWEET_EVAL_SENT",
    "BANKING77",
    "MRPC",
    "DEEHAN1866/WIC",
    "DGEM_FORMAT",
    "SICK",
    "STSB",
}

PAIR_SYMMETRIC_TASKS: Set[str] = {"MRPC", "STSB", "DEEHAN1866/WIC"}
PAIR_DIRECTED_TASKS: Set[str] = {"DGEM_FORMAT", "SICK"}


def compute_file_hash(file_path: str | Path) -> str:
    """Compute SHA256 checksum of a file for reproducibility tracking."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def map_scitail_label(value: Any) -> int:
    """Explicitly map SciTail labels to binary integer.
    
    Raises ValueError on unexpected labels to prevent silent data corruption.
    """
    label_str = str(value).strip().lower()
    if label_str in {"entails", "entailment", "1"}:
        return 1
    if label_str in {"neutral", "no entailment", "not_entailment", "0"}:
        return 0
    raise ValueError(f"Unexpected SciTail label: {value!r}")


def validate_training_csv(csv_path: str | Path) -> Dict[str, Any]:
    """Validate the pooled contrastive training CSV (final_data.csv)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)

    # Check required columns
    required_cols = {"text", "source"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing_cols)}")

    # Check non-null text and sources
    if df["text"].isnull().any():
        null_count = int(df["text"].isnull().sum())
        raise ValueError(f"Training CSV contains {null_count} null text entries.")
    if df["source"].isnull().any():
        null_count = int(df["source"].isnull().sum())
        raise ValueError(f"Training CSV contains {null_count} null source entries.")

    # Check source distribution
    present_sources = set(df["source"].unique())
    unknown_sources = present_sources - EXPECTED_SOURCES
    if unknown_sources:
        raise ValueError(f"Training CSV contains unexpected sources: {sorted(unknown_sources)}")

    missing_sources = EXPECTED_SOURCES - present_sources
    if missing_sources:
        raise ValueError(f"Training CSV is missing expected sources: {sorted(missing_sources)}")

    source_counts = df["source"].value_counts().to_dict()

    return {
        "file_path": str(path),
        "sha256": compute_file_hash(path),
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "source_counts": source_counts,
        "is_valid": True,
    }


def validate_eval_csv(
    csv_path: str | Path,
    task_type: str,
    task_name: str,
) -> Dict[str, Any]:
    """Validate an evaluation dataset split."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found for {task_name}: {path}")

    df = pd.read_csv(path)

    if task_type == "single":
        required = {"text", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} ({task_name}) missing columns: {sorted(missing)}")
        if df["text"].isnull().any():
            raise ValueError(f"{path} has null texts")
        if df["label"].isnull().any():
            raise ValueError(f"{path} has null labels")

    elif task_type in ("pair", "pair_regression"):
        required = {"sentence1", "sentence2", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} ({task_name}) missing columns: {sorted(missing)}")
        if df["sentence1"].isnull().any() or df["sentence2"].isnull().any():
            raise ValueError(f"{path} has null sentences")
        if df["label"].isnull().any():
            raise ValueError(f"{path} has null labels")

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return {
        "file_path": str(path),
        "sha256": compute_file_hash(path),
        "num_samples": len(df),
        "columns": df.columns.tolist(),
    }


def validate_probe_class_balance(csv_path: str | Path, min_samples_per_class: int = 3) -> Dict[int, int]:
    """Verify that every class in a classification dataset has at least min_samples_per_class."""
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        return {}
    class_counts = df["label"].value_counts().to_dict()
    low_classes = {cls: count for cls, count in class_counts.items() if count < min_samples_per_class}
    if low_classes:
        raise ValueError(
            f"{csv_path} has classes with fewer than {min_samples_per_class} samples: {low_classes}. "
            "Cannot perform 3-fold StratifiedKFold validation."
        )
    return class_counts


def check_cross_split_leakage(
    train_csv: str | Path,
    test_csv: str | Path,
    task_name: str,
    task_type: str = "single",
) -> Dict[str, Any]:
    """Check for identical text or task-aware reversed pairs between train and test splits."""
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    leakage_report: Dict[str, Any] = {
        "task_name": task_name,
        "exact_duplicates": 0,
        "reversed_duplicates": 0,
        "is_leakage": False,
        "informational_reversals": 0,
    }

    if task_type == "single":
        train_texts = set(train_df["text"].astype(str).str.strip())
        test_texts = set(test_df["text"].astype(str).str.strip())
        overlap = train_texts.intersection(test_texts)
        leakage_report["exact_duplicates"] = len(overlap)
        leakage_report["is_leakage"] = len(overlap) > 0

    elif task_type in ("pair", "pair_regression"):
        train_pairs = set(zip(
            train_df["sentence1"].astype(str).str.strip(),
            train_df["sentence2"].astype(str).str.strip()
        ))
        test_pairs = set(zip(
            test_df["sentence1"].astype(str).str.strip(),
            test_df["sentence2"].astype(str).str.strip()
        ))

        # 1. Exact pair matches
        exact_overlap = train_pairs.intersection(test_pairs)
        leakage_report["exact_duplicates"] = len(exact_overlap)

        # 2. Reversed pair matches
        test_reversed = {(s2, s1) for (s1, s2) in test_pairs}
        reversed_overlap = train_pairs.intersection(test_reversed)

        if task_name in PAIR_SYMMETRIC_TASKS:
            leakage_report["reversed_duplicates"] = len(reversed_overlap)
            leakage_report["is_leakage"] = (len(exact_overlap) > 0) or (len(reversed_overlap) > 0)
        else:
            leakage_report["informational_reversals"] = len(reversed_overlap)
            leakage_report["is_leakage"] = len(exact_overlap) > 0

    return leakage_report


def check_train_corpus_overlap(
    corpus_csv: str | Path,
    eval_test_csv: str | Path,
    task_name: str,
    task_type: str = "single",
) -> int:
    """Check if any evaluation test sentence appears in the contrastive training corpus."""
    corpus_df = pd.read_csv(corpus_csv)
    eval_df = pd.read_csv(eval_test_csv)

    corpus_texts = set(corpus_df["text"].astype(str).str.strip())

    if task_type == "single":
        test_texts = set(eval_df["text"].astype(str).str.strip())
    else:
        test_texts = set(eval_df["sentence1"].astype(str).str.strip()).union(
            set(eval_df["sentence2"].astype(str).str.strip())
        )

    overlap = corpus_texts.intersection(test_texts)
    return len(overlap)
