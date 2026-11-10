"""Paired bootstrap of the macro-F1 difference between two models, per (layer, width).

Both models were probed on the same 3,432 test rows, so the comparison is paired:
each resample draws row indices with replacement and scores BOTH models on the
same rows. One fixed index matrix (--n-resamples x n_rows, from --seed) is
shared by every (layer, width) cell, so all cells are judged on identical
resamples.

Inputs are the prediction files written by probe_grid.py:
    experiments/layerwise_probe/results/{a}_preds.npz and {b}_preds.npz
with keys L{layer}_d{dim}, P_d{dim} (pooler_output) and labels.

Output: experiments/layerwise_probe/results/bootstrap_{a}_vs_{b}.csv with, per cell, macro-F1 of
a and b, the observed difference a minus b, the 95% percentile interval of the
resampled differences, and whether that interval excludes zero. F1 values are
in points (x100).

Example, from the repository root:
    python experiments/layerwise_probe/bootstrap.py --a mipic_s42 --b mrl_s42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.layerwise_probe.probe_grid import DIMS, ROWS, prediction_key  # noqa: E402


def macro_f1_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Macro-F1 of each row of 2D (resamples x rows) label arrays.

    Same definition as sklearn's f1_score(average="macro"): per-class
    2TP / (2TP + FP + FN), 0 when that denominator is 0, averaged over
    ``classes``.
    """
    scores = []
    for c in classes:
        true_c = y_true == c
        pred_c = y_pred == c
        tp = np.sum(true_c & pred_c, axis=-1)
        fp = np.sum(~true_c & pred_c, axis=-1)
        fn = np.sum(true_c & ~pred_c, axis=-1)
        denom = 2 * tp + fp + fn
        scores.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(scores, axis=0)


def load_preds(results_dir: Path, name: str):
    path = results_dir / f"{name}_preds.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run probe_grid.py --name {name} first")
    return np.load(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap of macro-F1(a) - macro-F1(b) per (layer, width).")
    parser.add_argument("--a", required=True, help="First model's preds name, e.g. mipic_s42.")
    parser.add_argument("--b", required=True, help="Second model's preds name, e.g. mrl_s42 or base.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    preds_a = load_preds(args.results_dir, args.a)
    preds_b = load_preds(args.results_dir, args.b)
    labels = preds_a["labels"]
    if not np.array_equal(labels, preds_b["labels"]):
        raise ValueError(f"{args.a} and {args.b} were scored on different test labels; a paired bootstrap is invalid")

    n = len(labels)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_resamples, n))
    y_boot = labels[idx]

    keys = [(row, dim) for row in ROWS for dim in DIMS]
    classes = np.unique(np.concatenate(
        [labels] + [preds_a[prediction_key(r, d)] for r, d in keys] + [preds_b[prediction_key(r, d)] for r, d in keys]
    ))

    out = []
    for row, dim in keys:
        key = prediction_key(row, dim)
        pa, pb = preds_a[key], preds_b[key]
        f1_a = float(macro_f1_rows(labels, pa, classes))
        f1_b = float(macro_f1_rows(labels, pb, classes))
        boot = macro_f1_rows(y_boot, pa[idx], classes) - macro_f1_rows(y_boot, pb[idx], classes)
        low, high = np.percentile(boot, [2.5, 97.5])
        out.append({
            "layer": str(row),
            "dim": dim,
            "f1_a": f1_a * 100,
            "f1_b": f1_b * 100,
            "diff": (f1_a - f1_b) * 100,
            "ci_low": low * 100,
            "ci_high": high * 100,
            "excludes_zero": bool(low > 0 or high < 0),
            "boot_se": float(np.std(boot, ddof=1)) * 100,
        })

    table = pd.DataFrame(out)
    table.insert(0, "b", args.b)
    table.insert(0, "a", args.a)
    table["n_resamples"] = args.n_resamples
    table["seed"] = args.seed
    out_path = args.results_dir / f"bootstrap_{args.a}_vs_{args.b}.csv"
    table.to_csv(out_path, index=False)

    show = table.copy()
    show["cell"] = show.apply(
        lambda r: f"{r['diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]{'*' if r['excludes_zero'] else ''}", axis=1
    )
    pivot = show.pivot(index="layer", columns="dim", values="cell").reindex([str(r) for r in ROWS])
    print(f"{args.a} minus {args.b}, macro-F1 points, 95% CI from {args.n_resamples} paired resamples "
          f"(seed {args.seed}); * = CI excludes 0")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(pivot.to_string())
    print(f"\n{int(table['excludes_zero'].sum())} of {len(table)} cells exclude zero")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
