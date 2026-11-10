"""Paired bootstrap for the small-size tuned probe cells (Finding 1, check 6).

probe_tuned.py fits the scaled, CV-tuned probe on 5 subsets at each small
probe size but keeps only the full-size test predictions. This script refits
the small sizes (same subsets, same fit_tuned, so the F1 values must match
probe_tuned_{tag}.csv exactly, which is checked), saves the test predictions,
and runs a paired bootstrap of model minus base per (size, width):

    in each resample (the same resampled test rows for both models), each
    model's macro-F1 is averaged over its 5 subset fits, and the difference
    model minus base is taken.

1000 resamples of the test rows from default_rng(0), one index matrix shared
by every cell, model and subset, as in bootstrap.py.

Outputs, in layerwise_probe/results/:
    probe_tuned_{tag}_small_preds.npz        test predictions, key {model}_n{size}_r{rep}_d{dim}
    probe_tuned_{tag}_small_bootstrap.csv    one row per (model, size, dim)

Example, from the repository root:
    python layerwise_probe/probe_tuned_bootstrap.py --tag s1 --models base mrl_s1 mrl_pic_s1 mipic_s1 mrl_sia_s1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from layerwise_probe.bootstrap import macro_f1_rows  # noqa: E402
from layerwise_probe.probe_size import load_pooler, subset_indices  # noqa: E402
from layerwise_probe.probe_tuned import fit_tuned  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap of the small-size tuned probe cells.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reference", default="base")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 1000, 3000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--seed", type=int, default=0, help="Subset seed; must match probe_size.py and probe_tuned.py.")
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "results")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    data = {m: load_pooler(args.features_dir, m) for m in args.models}
    y_train, y_test = data[args.models[0]][1], data[args.models[0]][3]
    for m, (_, y_tr, _, y_te) in data.items():
        if not (np.array_equal(y_tr, y_train) and np.array_equal(y_te, y_test)):
            raise ValueError(f"{m} has different labels")
    n_train = len(y_train)
    subsets = {(s, r): subset_indices(n_train, s, r, args.seed) for s in args.sizes for r in range(args.reps)}

    jobs = [(m, s, r, d) for m in args.models for s in args.sizes for r in range(args.reps) for d in args.dims]
    jobs.sort(key=lambda j: -(j[1] * j[3]))

    def run(m, s, r, d):
        x_tr, y_tr, x_te, y_te = data[m]
        idx = subsets[(s, r)]
        out = fit_tuned(x_tr[idx], y_tr[idx], x_te, y_te, d)
        return m, s, r, d, out["predictions"], out["f1"]

    start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=1)(delayed(run)(*j) for j in jobs)
    print(f"{len(jobs)} refits in {time.perf_counter() - start:.0f}s")

    # The refits must reproduce probe_tuned.py's numbers exactly.
    saved = pd.read_csv(args.out_dir / f"probe_tuned_{args.tag}.csv").set_index(["model", "size", "rep", "dim"])["f1"]
    worst = max(abs(f1 - saved.loc[(m, s, r, d)]) for m, s, r, d, _, f1 in results)
    if worst > 1e-9:
        raise ValueError(f"refit F1 differs from probe_tuned_{args.tag}.csv by up to {worst:.2e}")
    print(f"refits reproduce probe_tuned_{args.tag}.csv exactly (max diff {worst:.1e})")

    preds = {f"{m}_n{s}_r{r}_d{d}": p for m, s, r, d, p, _ in results}
    np.savez_compressed(args.out_dir / f"probe_tuned_{args.tag}_small_preds.npz", labels=y_test, **preds)

    n = len(y_test)
    idx = np.random.default_rng(args.boot_seed).integers(0, n, size=(args.n_resamples, n))
    y_boot = y_test[idx]
    classes = np.unique(y_test)

    def mean_f1(model, size, dim):
        observed = np.mean([macro_f1_rows(y_test, preds[f"{model}_n{size}_r{r}_d{dim}"], classes)
                            for r in range(args.reps)]) * 100
        boot = np.mean([macro_f1_rows(y_boot, preds[f"{model}_n{size}_r{r}_d{dim}"][idx], classes)
                        for r in range(args.reps)], axis=0) * 100
        return float(observed), boot

    rows = []
    for s in args.sizes:
        for d in args.dims:
            ref_obs, ref_boot = mean_f1(args.reference, s, d)
            for m in args.models:
                if m == args.reference:
                    continue
                obs, boot = mean_f1(m, s, d)
                diff = boot - ref_boot
                low, high = np.percentile(diff, [2.5, 97.5])
                rows.append({"model": m, "reference": args.reference, "size": s, "dim": d,
                             "f1_model": obs, "f1_reference": ref_obs, "diff": obs - ref_obs,
                             "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0),
                             "model_above": bool(low > 0), "reference_above": bool(high < 0),
                             "boot_se": float(np.std(diff, ddof=1)), "n_reps": args.reps,
                             "n_resamples": args.n_resamples, "seed": args.boot_seed})
    table = pd.DataFrame(rows)
    out = args.out_dir / f"probe_tuned_{args.tag}_small_bootstrap.csv"
    table.to_csv(out, index=False)
    print(f"{args.tag}: model above in {int(table['model_above'].sum())}, reference above in "
          f"{int(table['reference_above'].sum())}, of {len(table)} cells")
    with pd.option_context("display.width", 200):
        print(table[["model", "size", "dim", "diff", "ci_low", "ci_high", "excludes_zero"]].round(2).to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
