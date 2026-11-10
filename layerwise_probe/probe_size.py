"""Probe-size sensitivity of the pooler probe (Finding 1).

Refits the probe_grid.py probe (LogisticRegression(random_state=42,
max_iter=1000), no scaling, one BLAS thread) on the pooler_output prefix using
only N training rows, for N in --sizes, with --reps random subsets per size,
plus the full training split. The subsets depend only on (size, rep) and are
drawn without replacement from default_rng(--seed + rep) after a per-size
offset, so every model is fitted on exactly the same rows. Test rows are
always the full test split.

Outputs, in layerwise_probe/results/:
    probe_size_{tag}.csv           one row per (model, size, rep, dim)
    probe_size_{tag}_summary.csv   per (model, size, dim): mean and SD of F1
                                   over reps, and the paired difference to
                                   --reference (mean, min, max over reps)

Example, from the repository root:
    python layerwise_probe/probe_size.py --models base mrl_s42 mrl_pic_s42 mipic_s42 mrl_sia_s42
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

from layerwise_probe.probe_grid import fit_one  # noqa: E402


def subset_indices(n_train: int, size: int, rep: int, seed: int) -> np.ndarray:
    """Training rows for one (size, rep). Independent of the model."""
    if size >= n_train:
        return np.arange(n_train)
    rng = np.random.default_rng(seed + 1000 * size + rep)
    return np.sort(rng.choice(n_train, size=size, replace=False))


def load_pooler(features_dir: Path, name: str):
    folder = features_dir / name
    x_train = np.load(folder / "train_pooler.npy").astype(np.float32)
    x_test = np.load(folder / "test_pooler.npy").astype(np.float32)
    y_train = np.load(folder / "train_labels.npy")
    y_test = np.load(folder / "test_labels.npy")
    return x_train, y_train, x_test, y_test


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pooler probe F1 as a function of probe training size.")
    parser.add_argument("--models", nargs="+", required=True, help="Feature folder names; the first is not special.")
    parser.add_argument("--reference", default="base", help="Model the paired differences are taken against.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 500, 1000, 3000, 10000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="s42")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "results")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    data = {m: load_pooler(args.features_dir, m) for m in args.models}
    y_train_ref, y_test_ref = data[args.models[0]][1], data[args.models[0]][3]
    for m, (_, y_tr, _, y_te) in data.items():
        if not (np.array_equal(y_tr, y_train_ref) and np.array_equal(y_te, y_test_ref)):
            raise ValueError(f"{m} has different train or test labels; subsets would not be shared")
    n_train = len(y_train_ref)

    plan = [(size, rep) for size in args.sizes for rep in range(args.reps)] + [(n_train, 0)]
    subsets = {(size, rep): subset_indices(n_train, size, rep, args.seed) for size, rep in plan}
    for (size, rep), idx in subsets.items():
        if len(np.unique(y_train_ref[idx])) < len(np.unique(y_train_ref)):
            raise ValueError(f"subset size {size} rep {rep} is missing a class")

    for m, (x_tr, _, _, _) in data.items():
        print(f"{m}: pooler feature scale, mean per-coordinate SD {x_tr.std(axis=0).mean():.5f}")

    jobs = [(m, size, rep, dim) for m in args.models for size, rep in plan for dim in args.dims]
    # Largest fits first so the slowest jobs do not start last.
    jobs.sort(key=lambda j: -(j[1] * j[3]))
    print(f"{len(jobs)} fits ({len(args.models)} models x {len(plan)} subsets x {len(args.dims)} widths)")

    def run(m, size, rep, dim):
        x_tr, y_tr, x_te, y_te = data[m]
        idx = subsets[(size, rep)]
        r = fit_one("pooler", dim, x_tr[idx], y_tr[idx], x_te, y_te)
        return {"model": m, "size": size, "rep": rep, "dim": dim, "f1": r["f1"] * 100,
                "accuracy": r["accuracy"] * 100, "converged": r["converged"], "n_iter": r["n_iter"]}

    start = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=2)(delayed(run)(*j) for j in jobs)
    print(f"done in {time.perf_counter() - start:.0f}s")

    table = pd.DataFrame(rows).sort_values(["model", "size", "rep", "dim"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"probe_size_{args.tag}.csv", index=False)
    if not table["converged"].all():
        bad = table[~table["converged"]]
        print(f"WARNING: {len(bad)} fits did not converge:\n{bad[['model', 'size', 'rep', 'dim']].to_string(index=False)}")

    ref = table[table["model"] == args.reference].set_index(["size", "rep", "dim"])["f1"]
    table["diff_vs_ref"] = table["f1"].to_numpy() - ref.loc[list(zip(table["size"], table["rep"], table["dim"]))].to_numpy()
    summary = (table.groupby(["model", "size", "dim"])
               .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), n_reps=("f1", "size"),
                    diff_mean=("diff_vs_ref", "mean"), diff_min=("diff_vs_ref", "min"),
                    diff_max=("diff_vs_ref", "max"), all_converged=("converged", "all"))
               .reset_index())
    summary.insert(1, "reference", args.reference)
    summary.to_csv(args.out_dir / f"probe_size_{args.tag}_summary.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        for dim in args.dims:
            s = summary[summary["dim"] == dim]
            print(f"\nPooler {dim}D, mean F1 over reps (SD):")
            print(s.assign(cell=s.apply(lambda r: f"{r.f1_mean:.2f} ({0 if pd.isna(r.f1_sd) else r.f1_sd:.2f})", axis=1))
                  .pivot(index="size", columns="model", values="cell")[args.models].to_string())
    print(f"\nSaved {args.out_dir / f'probe_size_{args.tag}.csv'} and the _summary.csv")


if __name__ == "__main__":
    main()
