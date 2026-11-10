"""Tuned pooler probe at several probe sizes (Finding 1, check 6).

Same subsets as probe_size.py (probe_size.subset_indices, same --seed), but the
probe is StandardScaler + LogisticRegression(random_state=42, max_iter=1000)
with C chosen by 5-fold stratified cross-validation (macro-F1) on the probe
training rows only. The scaler sits inside the pipeline, so in every fold it is
fitted on that fold's training part, and the final refit scales with statistics
from the probe training rows only. Test rows are always the full test split.

Outputs, in layerwise_probe/results/:
    probe_tuned_{tag}.csv           one row per (model, size, rep, dim), with chosen C
    probe_tuned_{tag}_summary.csv   per (model, size, dim): mean and SD of F1,
                                    paired difference to --reference, C counts

Example, from the repository root:
    python layerwise_probe/probe_tuned.py --models base mrl_s42 mrl_pic_s42 mipic_s42 mrl_sia_s42
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from layerwise_probe.probe_size import load_pooler, subset_indices  # noqa: E402

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CV_SEED = 0


def fit_tuned(x_train, y_train, x_test, y_test, dim: int) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    search = GridSearchCV(
        pipe, {"lr__C": list(C_GRID)}, scoring="f1_macro",
        cv=StratifiedKFold(5, shuffle=True, random_state=CV_SEED), n_jobs=1, refit=True,
    )
    x_fit = np.ascontiguousarray(x_train[:, :dim])
    x_eval = np.ascontiguousarray(x_test[:, :dim])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(x_fit, y_train)
        predictions = search.predict(x_eval)
    n_iter = int(search.best_estimator_.named_steps["lr"].n_iter_[0])
    return {
        "predictions": predictions.astype(np.int8),
        "f1": float(f1_score(y_test, predictions, average="macro")) * 100,
        "accuracy": float(accuracy_score(y_test, predictions)) * 100,
        "best_C": float(search.best_params_["lr__C"]),
        "cv_f1": float(search.best_score_) * 100,
        "n_iter": n_iter,
        "converged": n_iter < 1000,
    }


def bootstrap_full(preds: dict, labels: np.ndarray, models: List[str], reference: str, dims: List[int],
                   n_resamples: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of F1(model) - F1(reference) on the full-size fits, as in bootstrap.py."""
    from layerwise_probe.bootstrap import macro_f1_rows

    n = len(labels)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    y_boot = labels[idx]
    classes = np.unique(labels)
    out = []
    for m in models:
        if m == reference:
            continue
        for dim in dims:
            a, b = preds[f"{m}_d{dim}"], preds[f"{reference}_d{dim}"]
            diff = float(macro_f1_rows(labels, a, classes) - macro_f1_rows(labels, b, classes)) * 100
            boot = (macro_f1_rows(y_boot, a[idx], classes) - macro_f1_rows(y_boot, b[idx], classes)) * 100
            low, high = np.percentile(boot, [2.5, 97.5])
            out.append({"model": m, "reference": reference, "dim": dim, "diff": diff, "ci_low": low, "ci_high": high,
                        "excludes_zero": bool(low > 0 or high < 0), "n_resamples": n_resamples, "seed": seed})
    return pd.DataFrame(out)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Scaled, CV-tuned pooler probe across probe sizes.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--reference", default="base")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 1000, 3000])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--seed", type=int, default=0, help="Must match probe_size.py to reuse its subsets.")
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

    jobs = [(m, size, rep, dim) for m in args.models for size, rep in plan for dim in args.dims]
    jobs.sort(key=lambda j: -(j[1] * j[3]))
    print(f"{len(jobs)} tuned fits, each 5 folds x {len(C_GRID)} C values + refit")

    def run(m, size, rep, dim):
        x_tr, y_tr, x_te, y_te = data[m]
        idx = subsets[(size, rep)]
        return {"model": m, "size": size, "rep": rep, "dim": dim, **fit_tuned(x_tr[idx], y_tr[idx], x_te, y_te, dim)}

    start = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=2)(delayed(run)(*j) for j in jobs)
    print(f"done in {time.perf_counter() - start:.0f}s")

    # Full-size test predictions, for the paired bootstrap against --reference below.
    full_preds = {f"{r['model']}_d{r['dim']}": r.pop("predictions") for r in rows if r["size"] == n_train}
    for r in rows:
        r.pop("predictions", None)
    table = pd.DataFrame(rows).sort_values(["model", "size", "rep", "dim"]).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"probe_tuned_{args.tag}.csv", index=False)
    np.savez_compressed(args.out_dir / f"probe_tuned_{args.tag}_preds.npz", labels=y_test_ref, **full_preds)
    boot = bootstrap_full(full_preds, y_test_ref, args.models, args.reference, args.dims)
    boot.to_csv(args.out_dir / f"probe_tuned_{args.tag}_bootstrap.csv", index=False)
    if not table["converged"].all():
        print(f"WARNING: {int((~table['converged']).sum())} final fits hit max_iter")

    ref = table[table["model"] == args.reference].set_index(["size", "rep", "dim"])["f1"]
    table["diff_vs_ref"] = table["f1"].to_numpy() - ref.loc[list(zip(table["size"], table["rep"], table["dim"]))].to_numpy()
    summary = (table.groupby(["model", "size", "dim"])
               .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"), n_reps=("f1", "size"),
                    diff_mean=("diff_vs_ref", "mean"), diff_min=("diff_vs_ref", "min"),
                    diff_max=("diff_vs_ref", "max"),
                    chosen_C=("best_C", lambda c: " ".join(f"{v:g}" for v in sorted(c))),
                    all_converged=("converged", "all"))
               .reset_index())
    summary.insert(1, "reference", args.reference)
    summary.to_csv(args.out_dir / f"probe_tuned_{args.tag}_summary.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        for dim in args.dims:
            s = summary[summary["dim"] == dim]
            print(f"\nTuned pooler {dim}D, mean F1 over reps (SD) [chosen C]:")
            print(s.assign(cell=s.apply(lambda r: f"{r.f1_mean:.2f} ({0 if pd.isna(r.f1_sd) else r.f1_sd:.2f}) [{r.chosen_C}]", axis=1))
                  .pivot(index="size", columns="model", values="cell")[args.models].to_string())
    print(f"\nSaved {args.out_dir / f'probe_tuned_{args.tag}.csv'} and the _summary.csv")


if __name__ == "__main__":
    main()
