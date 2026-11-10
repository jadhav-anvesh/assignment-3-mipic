"""Is the prefix special? Probe the first d coordinates against other d-coordinate subsets.

Tests H1/H2: if Matryoshka training packs the most useful information into the
leading coordinates, a probe on [0:d] should beat a probe on any other d
coordinates. Untrained TinyBERT is the control, where no such ordering is expected.

For the pooler row and layers 1-6 at widths 16, 32, 64, 128 it compares:
    prefix      coordinates [0:d]
    next_block  coordinates [d:2d]
    last_block  coordinates [768-d:768]
    random      --n-random random d-subsets, seeds 0..n-1, identical for every
                layer and every model (saved to random_subsets.json)

Probe settings are those of probe_grid.py (and so of evaluate.py).

Example, from the repository root:
    python layerwise_probe/subset_probe.py --name base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from layerwise_probe.probe_grid import ROWS, fit_one, load_features, prediction_key  # noqa: E402

SUBSET_DIMS = (16, 32, 64, 128)
HIDDEN = 768
BLOCK_METHODS = ("prefix", "next_block", "last_block")


def block_indices(method: str, dim: int) -> np.ndarray:
    if method == "prefix":
        return np.arange(0, dim)
    if method == "next_block":
        return np.arange(dim, 2 * dim)
    if method == "last_block":
        return np.arange(HIDDEN - dim, HIDDEN)
    raise ValueError(method)


def make_random_subsets(n_random: int) -> Dict[str, Dict[str, List[int]]]:
    """{dim: {seed: sorted indices}}; depends only on (dim, seed), never on layer or model."""
    return {
        str(dim): {
            str(seed): sorted(np.random.default_rng(seed).choice(HIDDEN, size=dim, replace=False).tolist())
            for seed in range(n_random)
        }
        for dim in SUBSET_DIMS
    }


def load_or_save_subsets(path: Path, n_random: int) -> Dict[str, Dict[str, List[int]]]:
    subsets = make_random_subsets(n_random)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
        for dim, by_seed in subsets.items():
            for seed, indices in by_seed.items():
                if stored.get(dim, {}).get(seed, indices) != indices:
                    raise ValueError(f"{path} disagrees with regenerated subset dim={dim} seed={seed}")
        # Keep any extra seeds already on disk; add new ones.
        for dim, by_seed in subsets.items():
            stored.setdefault(dim, {}).update(by_seed)
        subsets_to_write = stored
    else:
        subsets_to_write = subsets
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(subsets_to_write, handle)
    return subsets


def fit_subset(row, dim, method, seed, indices, x_train, y_train, x_test, y_test):
    result = fit_one(row, dim, x_train[:, indices], y_train, x_test[:, indices], y_test)
    result.update({"method": method, "subset_seed": seed})
    return result


def check_prefix_against_grid(
    name: str, out_dir: Path, fits: pd.DataFrame, prefix_preds: Dict[str, np.ndarray]
) -> None:
    """The prefix fit is the same fit as probe_grid.py's cell, so it must reproduce it.

    Test predictions are compared exactly against {name}_preds.npz. F1 and accuracy
    are compared to 1e-12, because the CSV stores floats with one digit fewer than
    a full round trip needs (e.g. 0.41673798261659284 is written 0.4167379826165928),
    while a single flipped prediction moves F1 by about 1e-4.
    """
    grid_path = out_dir / f"{name}_grid.csv"
    preds_path = out_dir / f"{name}_preds.npz"
    for path in (grid_path, preds_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run probe_grid.py --name {name} first")
    grid = pd.read_csv(grid_path, dtype={"layer": str})
    stored_preds = np.load(preds_path)

    prefix = fits[fits["method"] == "prefix"]
    merged = prefix.merge(grid, on=["layer", "dim"], suffixes=("", "_grid"), how="left")
    if merged["f1_grid"].isna().any():
        raise ValueError("Some prefix cells have no matching row in the grid CSV")

    problems = []
    for r in merged.itertuples():
        key = prediction_key(int(r.layer) if r.layer != "pooler" else "pooler", r.dim)
        if not np.array_equal(prefix_preds[key], stored_preds[key]):
            flipped = int((prefix_preds[key] != stored_preds[key]).sum())
            problems.append(f"{key}: {flipped} test predictions differ")
        if abs(r.f1 - r.f1_grid) > 1e-12 or abs(r.accuracy - r.accuracy_grid) > 1e-12:
            problems.append(f"{key}: f1 {r.f1!r} vs {r.f1_grid!r}, accuracy {r.accuracy!r} vs {r.accuracy_grid!r}")
    if problems:
        raise ValueError(f"Prefix fits do not reproduce {grid_path.name}:\n  " + "\n  ".join(problems))
    print(f"Check passed: all {len(merged)} prefix fits reproduce {grid_path.name} "
          f"(identical test predictions in {preds_path.name})")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare the prefix with other coordinate subsets (H1/H2).")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base, mrl, mrl_pic, mipic.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "results")
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)

    (train, y_train), (test, y_test) = load_features(args.features_dir / args.name, None)
    subsets = load_or_save_subsets(args.out_dir / "random_subsets.json", args.n_random)

    jobs = []
    for row in ROWS:
        for dim in SUBSET_DIMS:
            for method in BLOCK_METHODS:
                jobs.append((row, dim, method, None, block_indices(method, dim)))
            for seed in range(args.n_random):
                jobs.append((row, dim, "random", seed, np.array(subsets[str(dim)][str(seed)])))
    jobs.sort(key=lambda job: -job[1])  # widest first for load balancing
    print(f"{args.name}: {len(jobs)} fits ({len(ROWS)} rows x {len(SUBSET_DIMS)} widths x "
          f"{len(BLOCK_METHODS) + args.n_random} subsets), n_jobs={args.n_jobs}")

    start = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_subset)(row, dim, method, seed, idx, train[row], y_train, test[row], y_test)
        for row, dim, method, seed, idx in jobs
    )
    total = time.perf_counter() - start

    fits = pd.DataFrame([
        {
            "model": args.name,
            "layer": str(r["layer"]),
            "dim": r["dim"],
            "method": r["method"],
            "subset_seed": r["subset_seed"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
            "converged": r["converged"],
        }
        for r in results
    ])
    order = {str(row): i for i, row in enumerate(ROWS)}
    method_order = {m: i for i, m in enumerate(BLOCK_METHODS + ("random",))}
    fits = fits.sort_values(
        by=["layer", "dim", "method", "subset_seed"],
        key=lambda s: s.map(order) if s.name == "layer" else s.map(method_order) if s.name == "method" else s,
    ).reset_index(drop=True)
    fits["subset_seed"] = fits["subset_seed"].astype("Int64")

    for r in fits[~fits["converged"]].itertuples():
        print(f"WARNING: layer {r.layer} dim {r.dim} {r.method} seed {r.subset_seed} did not converge")

    prefix_preds = {
        prediction_key(r["layer"], r["dim"]): r["predictions"] for r in results if r["method"] == "prefix"
    }
    check_prefix_against_grid(args.name, args.out_dir, fits, prefix_preds)

    rows = []
    for (layer, dim), group in fits.groupby(["layer", "dim"], sort=False):
        f1 = {m: float(group.loc[group["method"] == m, "f1"].iloc[0]) for m in BLOCK_METHODS}
        rand = group.loc[group["method"] == "random", "f1"]
        rows.append({
            "model": args.name,
            "layer": layer,
            "dim": dim,
            "prefix_f1": f1["prefix"],
            "next_block_f1": f1["next_block"],
            "last_block_f1": f1["last_block"],
            "random_mean_f1": float(rand.mean()),
            "random_std_f1": float(rand.std(ddof=1)) if len(rand) > 1 else float("nan"),
            "prefix_minus_random": f1["prefix"] - float(rand.mean()),
        })
    summary = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = args.out_dir / f"{args.name}_subsets.csv"
    summary_path = args.out_dir / f"{args.name}_subsets_summary.csv"
    fits.to_csv(fits_path, index=False)
    summary.to_csv(summary_path, index=False)

    row_labels = [str(r) for r in ROWS]
    pivot = (summary.pivot(index="layer", columns="dim", values="prefix_minus_random") * 100).reindex(row_labels)
    std = (summary.pivot(index="layer", columns="dim", values="random_std_f1") * 100).reindex(row_labels)
    print(f"\nprefix_minus_random, macro-F1 points, for {args.name}:")
    print(pivot.round(2).to_string())
    print(f"\nrandom-subset std, macro-F1 points (n={args.n_random} subsets):")
    print(std.round(2).to_string())
    print(f"\n{len(fits)} fits in {total:.1f}s wall time")
    print(f"Saved {fits_path}, {summary_path} and {args.out_dir / 'random_subsets.json'}")


if __name__ == "__main__":
    main()
