"""Paired bootstrap of STS-B Spearman differences, model minus base (Finding 1 scope table).

Encodes the STS-B test pairs exactly as sts_eval.py does (same loader, overlap
filter, extractor and prefix cosine), then for each cell compares a trained
model with base on the SAME resampled pairs:

    diff = Spearman(cos_model, gold) - Spearman(cos_base, gold)

over --n-resamples resamples of pair indices from default_rng(--seed), one index
matrix shared by every cell and model. Spearman on a resample is the Pearson
correlation of average ranks, which equals scipy's spearmanr with ties.

Cells: the pooler row at every width, and raw [CLS] layer 6 at 768D (PIC
checkpoint (6,768)). Per-pair cosines are cached in
results/sts_cosines_{name}{_clean}.npz so base is encoded once. Observed
Spearman values are checked against results/{name}_sts{_clean}.csv.

Outputs results/bootstrap_sts_{model}_vs_{reference}{_clean}.csv, in Spearman x 100.

Example, from the repository root:
    python experiments/layerwise_probe/bootstrap_sts.py --suffix _s1 --drop-overlap
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

from experiments.layerwise_probe.probe_grid import DIMS  # noqa: E402

CELLS = [("pooler", d) for d in DIMS] + [(6, 768)]
MODELS = ("mrl", "mrl_pic", "mipic", "mrl_sia")


def cell_key(row, dim) -> str:
    return f"P_d{dim}" if row == "pooler" else f"L{row}_d{dim}"


def cosines(name: str, checkpoint: Optional[Path], drop_overlap: bool, results_dir: Path):
    """Per-pair cosines for every cell, plus gold scores; cached on disk."""
    tag = "_clean" if drop_overlap else ""
    cache = results_dir / f"sts_cosines_{name}{tag}.npz"
    if cache.exists():
        data = np.load(cache)
        return {k: data[k] for k in data.files if k != "gold"}, data["gold"]

    import torch
    from experiments.layerwise_probe.extract_cls import extract_split
    from experiments.layerwise_probe.load_encoder import load_encoder
    from experiments.layerwise_probe.sts_eval import find_overlap, load_stsb_test, prefix_cosine

    pairs = load_stsb_test()
    affected = find_overlap(pairs, results_dir)
    if drop_overlap:
        pairs = pairs.loc[~affected].reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer = load_encoder(checkpoint, device=device)
    first = extract_split(encoder, tokenizer, pairs["sentence1"].tolist(), 64, 256, desc=f"{name}/sentence1")
    second = extract_split(encoder, tokenizer, pairs["sentence2"].tolist(), 64, 256, desc=f"{name}/sentence2")
    out = {cell_key(r, d): prefix_cosine(first[r][:, :d], second[r][:, :d]) for r, d in CELLS}
    gold = pairs["score"].to_numpy()
    np.savez_compressed(cache, gold=gold, **out)
    del encoder
    return out, gold


def check_observed(name: str, cos: dict, gold: np.ndarray, drop_overlap: bool, results_dir: Path) -> None:
    table = pd.read_csv(results_dir / f"{name}_sts{'_clean' if drop_overlap else ''}.csv")
    table["layer"] = table["layer"].astype(str)
    for r, d in CELLS:
        saved = float(table[(table["layer"] == str(r)) & (table["dim"] == d)]["spearman"].iloc[0])
        mine = float(spearmanr(cos[cell_key(r, d)], gold).statistic)
        if abs(saved - mine) > 1e-6:
            raise ValueError(f"{name} {r} {d}: Spearman {mine:.6f} does not match saved {saved:.6f}")


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman of each row of 2D (resamples x pairs) arrays."""
    rx = rankdata(x, axis=1)
    ry = rankdata(y, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap of STS-B Spearman, model minus base.")
    parser.add_argument("--suffix", required=True, help="Seed suffix, e.g. _s42, _s1, _s2.")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--reference", default="base")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Checkpoint folder; default experiments/layerwise_probe/runs/seed<N>/checkpoints from --suffix.")
    parser.add_argument("--drop-overlap", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir or REPO_ROOT / "experiments" / "layerwise_probe" / "runs" / f"seed{args.suffix.lstrip('_s')}" / "checkpoints"
    ref_cos, gold = cosines(args.reference, None, args.drop_overlap, args.results_dir)
    check_observed(args.reference, ref_cos, gold, args.drop_overlap, args.results_dir)

    n = len(gold)
    idx = np.random.default_rng(args.seed).integers(0, n, size=(args.n_resamples, n))
    gold_boot = gold[idx]
    ref_boot = {k: spearman_rows(v[idx], gold_boot) for k, v in ref_cos.items()}
    tag = "_clean" if args.drop_overlap else ""

    for m in args.models:
        name = f"{m}{args.suffix}"
        cos, gold_m = cosines(name, runs_dir / m / "best.pt", args.drop_overlap, args.results_dir)
        if not np.array_equal(gold, gold_m):
            raise ValueError(f"{name} was scored on different pairs than {args.reference}")
        check_observed(name, cos, gold, args.drop_overlap, args.results_dir)
        rows = []
        for r, d in CELLS:
            k = cell_key(r, d)
            a = float(spearmanr(cos[k], gold).statistic) * 100
            b = float(spearmanr(ref_cos[k], gold).statistic) * 100
            boot = (spearman_rows(cos[k][idx], gold_boot) - ref_boot[k]) * 100
            low, high = np.percentile(boot, [2.5, 97.5])
            rows.append({"model": name, "reference": args.reference, "layer": str(r), "dim": d,
                         "spearman_model": a, "spearman_reference": b, "diff": a - b,
                         "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0),
                         "boot_se": float(np.std(boot, ddof=1)), "n_pairs": n,
                         "n_resamples": args.n_resamples, "seed": args.seed})
        table = pd.DataFrame(rows)
        out = args.results_dir / f"bootstrap_sts_{name}_vs_{args.reference}{tag}.csv"
        table.to_csv(out, index=False)
        cells = ", ".join(f"{r['layer']}/{r['dim']} {r['diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
                          for _, r in table.iterrows())
        print(f"{name} - {args.reference}: {cells}")


if __name__ == "__main__":
    main()
