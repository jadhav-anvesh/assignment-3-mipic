"""Fit the evaluate.py logistic-regression probe on every (layer, width) [CLS] prefix.

Reads the features written by extract_cls.py and, for layers 1-6 plus the
pooler_output row and widths 16..768, fits on train[:, :d] and scores on
test[:, :d]. Probe settings are copied from evaluate.evaluate_tweeteval:
LogisticRegression(random_state=42, max_iter=1000, verbose=0) with sklearn
defaults otherwise (lbfgs, L2, C=1.0) and no feature scaling.

The pooler row is the vector evaluate.py probes, so --check-readme can compare it
against the README table stored in expected_readme.json.

Example, from the repository root:
    python h11_probe/probe_grid.py --name base
    python h11_probe/probe_grid.py --name mipic --check-readme
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

LAYERS = (1, 2, 3, 4, 5, 6)
# Row keys: transformer layers 1-6, then "pooler" (always shown last).
ROWS: Tuple[Union[int, str], ...] = LAYERS + ("pooler",)
DIMS = (16, 32, 64, 128, 256, 512, 768)
PIC_CHECKPOINTS = {(1, 16), (2, 32), (3, 64), (4, 256), (5, 512), (6, 768)}
EXPECTED_README = REPO_ROOT / "h11_probe" / "expected_readme.json"
README_TOLERANCE = 1.0

# Same values as evaluate.evaluate_tweeteval (random_state is hardcoded to 42
# there; max_iter is MRLEvaluationConfig.max_iter, which run_experiments sets to 1000).
PROBE_RANDOM_STATE = 42
PROBE_MAX_ITER = 1000


def feature_filename(split: str, row: Union[int, str]) -> str:
    return f"{split}_pooler.npy" if row == "pooler" else f"{split}_layer{row}.npy"


def prediction_key(row: Union[int, str], dim: int) -> str:
    return f"P_d{dim}" if row == "pooler" else f"L{row}_d{dim}"


def load_features(feature_dir: Path, limit: Optional[int]):
    def split(name: str):
        labels = np.load(feature_dir / f"{name}_labels.npy")
        rows = {
            row: np.load(feature_dir / feature_filename(name, row)).astype(np.float32)
            for row in ROWS
        }
        if limit is not None:
            labels = labels[:limit]
            rows = {row: array[:limit] for row, array in rows.items()}
        return rows, labels

    return split("train"), split("test")


def fit_one(row: Union[int, str], dim: int, x_train, y_train, x_test, y_test) -> Dict[str, object]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    start = time.perf_counter()
    classifier = LogisticRegression(
        random_state=PROBE_RANDOM_STATE, max_iter=PROBE_MAX_ITER, verbose=0
    )
    with warnings.catch_warnings():
        # Convergence is recorded from n_iter_ below; keep worker output quiet.
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(x_train[:, :dim], y_train)
    predictions = classifier.predict(x_test[:, :dim])
    n_iter = int(classifier.n_iter_[0])
    return {
        "layer": row,
        "dim": dim,
        "f1": float(f1_score(y_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "converged": n_iter < PROBE_MAX_ITER,
        "n_iter": n_iter,
        "fit_seconds": round(time.perf_counter() - start, 2),
        "predictions": predictions.astype(np.int8),
    }


def check_readme(name: str, grid: pd.DataFrame) -> None:
    with open(EXPECTED_README, "r", encoding="utf-8") as handle:
        expected = json.load(handle)["macro_f1"]
    if name not in expected:
        print(f"\n--check-readme: '{name}' is not in {EXPECTED_README.name} "
              f"(has: {', '.join(expected)}); skipping.")
        return

    pooler = grid[grid["layer"] == "pooler"].set_index("dim")["f1"] * 100
    print(f"\nPooler row vs README for {name} (macro-F1 %):")
    print(f"{'dim':>5} {'ours':>7} {'README':>7} {'diff':>7}")
    worst = 0.0
    for dim in DIMS:
        ours, theirs = float(pooler[dim]), float(expected[name][str(dim)])
        worst = max(worst, abs(ours - theirs))
        print(f"{dim:>5} {ours:7.2f} {theirs:7.2f} {ours - theirs:+7.2f}")
    verdict = "PASS" if worst < README_TOLERANCE else "WARN"
    print(f"{verdict}: largest |diff| = {worst:.2f} (tolerance {README_TOLERANCE:.1f})")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Probe every (layer, width) [CLS] prefix for H11.")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base, mrl, mrl_pic, mipic.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "results")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rows of each split.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--check-readme", action="store_true",
                        help="Compare the pooler row with the README table in expected_readme.json.")
    args = parser.parse_args(argv)

    (train, y_train), (test, y_test) = load_features(args.features_dir / args.name, args.limit)
    print(f"{args.name}: {len(y_train)} train rows, {len(y_test)} test rows, "
          f"{len(ROWS) * len(DIMS)} fits, n_jobs={args.n_jobs}")

    # Widest fits first so the slowest jobs do not start last.
    jobs = sorted(((row, dim) for row in ROWS for dim in DIMS), key=lambda job: -job[1])
    start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_one)(row, dim, train[row], y_train, test[row], y_test)
        for row, dim in jobs
    )
    total = time.perf_counter() - start

    results.sort(key=lambda r: (ROWS.index(r["layer"]), r["dim"]))
    for r in results:
        if not r["converged"]:
            print(f"WARNING: layer {r['layer']} dim {r['dim']} did not converge "
                  f"(n_iter={r['n_iter']} of max_iter={PROBE_MAX_ITER})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame([
        {
            "model": args.name,
            "layer": r["layer"],
            "dim": r["dim"],
            "is_pic_checkpoint": (r["layer"], r["dim"]) in PIC_CHECKPOINTS,
            "f1": r["f1"],
            "accuracy": r["accuracy"],
            "converged": r["converged"],
            "n_iter": r["n_iter"],
            "fit_seconds": r["fit_seconds"],
        }
        for r in results
    ])
    grid_path = args.out_dir / f"{args.name}_grid.csv"
    grid.to_csv(grid_path, index=False)

    preds = {prediction_key(r["layer"], r["dim"]): r["predictions"] for r in results}
    preds["labels"] = y_test
    preds_path = args.out_dir / f"{args.name}_preds.npz"
    np.savez_compressed(preds_path, **preds)

    pivot = (grid.pivot(index="layer", columns="dim", values="f1") * 100).reindex(list(ROWS))
    print(f"\nMacro-F1 (%) for {args.name}, layers x widths:")
    print(pivot.round(2).to_string())
    print(f"\n{len(results)} fits in {total:.1f}s wall time "
          f"({grid['fit_seconds'].sum():.1f}s summed across workers)")
    print(f"Saved {grid_path} and {preds_path}")

    if args.check_readme:
        check_readme(args.name, grid)


if __name__ == "__main__":
    main()
