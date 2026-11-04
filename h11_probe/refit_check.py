"""Refit a model's pooler row with today's code and compare with its saved predictions (Finding 1, check 2).

Loads features/{name}/*_pooler.npy, refits probe_grid.fit_one at every width,
and compares the test predictions with results/{name}_preds.npz and the F1
with results/{name}_grid.csv. Writes results/refit_check_{name}.csv.

Example, from the repository root:
    python h11_probe/refit_check.py --name base
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from h11_probe.probe_grid import DIMS, fit_one, prediction_key  # noqa: E402
from h11_probe.probe_size import load_pooler  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Refit the pooler row and compare with saved predictions.")
    parser.add_argument("--name", default="base")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "h11_probe" / "results")
    args = parser.parse_args(argv)

    x_tr, y_tr, x_te, y_te = load_pooler(args.features_dir, args.name)
    saved = np.load(args.results_dir / f"{args.name}_preds.npz")
    grid = pd.read_csv(args.results_dir / f"{args.name}_grid.csv")
    grid = grid[grid["layer"].astype(str) == "pooler"].set_index("dim")["f1"]

    rows = []
    for dim in DIMS:
        r = fit_one("pooler", dim, x_tr, y_tr, x_te, y_te)
        old = saved[prediction_key("pooler", dim)]
        rows.append({"name": args.name, "dim": dim, "f1_refit": r["f1"] * 100,
                     "f1_grid": float(grid.loc[dim]) * (100 if grid.max() <= 1 else 1),
                     "n_predictions_differ": int((r["predictions"] != old).sum())})
    table = pd.DataFrame(rows)
    out = args.results_dir / f"refit_check_{args.name}.csv"
    table.to_csv(out, index=False)
    print(table.round(2).to_string(index=False))
    print(f"identical predictions at every width: {bool((table['n_predictions_differ'] == 0).all())}\nSaved {out}")


if __name__ == "__main__":
    main()
