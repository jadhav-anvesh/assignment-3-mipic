"""Reverse pooler swap: trained h6 through base's pooler (Finding 9).

The other direction of pooler_swap.py. Each trained model's saved layer 6 [CLS]
features (features/{model}{tag}/*_layer6.npy) are passed through BASE
TinyBERT's pooler, tanh(W_base h6 + b_base). If the encoder carries the
change, this keeps the trained model's feature scale and probe F1; if the
trained pooler carried it, this falls back toward base.

For comparison each row also reports the model's own pooler features (trained
pooler on trained h6, recomputed from the checkpoint and checked against the
stored *_pooler.npy) and base's. Probe: probe_grid.fit_one, the default probe.

Outputs experiments/layerwise_probe/results/pooler_swap_reverse_{tag}.csv with one row per
(model, readout, dim): readout is "own" (trained W, trained h6), "base_pooler"
(base W, trained h6) or "base" (base W, base h6).

Example, from the repository root:
    python experiments/layerwise_probe/pooler_swap_reverse.py --tag s42
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
from joblib import Parallel, delayed  # noqa: E402

from experiments.layerwise_probe.load_encoder import load_encoder  # noqa: E402
from experiments.layerwise_probe.pooler_swap import MODELS, pool  # noqa: E402
from experiments.layerwise_probe.probe_grid import fit_one  # noqa: E402


def load_h6(folder: Path):
    return (np.load(folder / "train_layer6.npy").astype(np.float64),
            np.load(folder / "test_layer6.npy").astype(np.float64),
            np.load(folder / "train_labels.npy"), np.load(folder / "test_labels.npy"))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Probe trained layer 6 features through base's pooler.")
    parser.add_argument("--tag", required=True, help="Seed tag, e.g. s42, s1, s2.")
    parser.add_argument("--runs-dir", type=Path, default=None,
                        help="Checkpoint folder; default experiments/layerwise_probe/runs/seed<N>/checkpoints from --tag.")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results")
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir or REPO_ROOT / "experiments" / "layerwise_probe" / "runs" / f"seed{args.tag.lstrip('s')}" / "checkpoints"
    base_encoder, _ = load_encoder(None, device="cpu")
    bh_tr, bh_te, y_tr, y_te = load_h6(args.features_dir / "base")
    base_te = pool(base_encoder, bh_te)
    base_sd = base_te.std(axis=0).mean()

    feats = {("base", "base"): (pool(base_encoder, bh_tr), base_te)}
    for m in args.models:
        name = f"{m}_{args.tag}"
        folder = args.features_dir / name
        h_tr, h_te, my_tr, my_te = load_h6(folder)
        if not (np.array_equal(my_tr, y_tr) and np.array_equal(my_te, y_te)):
            raise ValueError(f"{name} has different labels from base")
        encoder, _ = load_encoder(runs_dir / m / "best.pt", device="cpu")
        own_te = pool(encoder, h_te)
        stored = np.load(folder / "test_pooler.npy").astype(np.float64)
        gap = np.abs(own_te - stored).max()
        if gap > 1e-4:
            raise ValueError(f"{name}: recomputed pooler differs from stored test_pooler.npy by {gap:.2e}")
        feats[(name, "own")] = (pool(encoder, h_tr), own_te)
        feats[(name, "base_pooler")] = (pool(base_encoder, h_tr), pool(base_encoder, h_te))
        print(f"{name}: recomputed own pooler matches stored features (max abs diff {gap:.1e})")

    stats = {}
    for key, (_, te) in feats.items():
        own = feats.get((key[0], "own"), feats[("base", "base")])[1]
        cos_own = (te * own).sum(1) / (np.linalg.norm(te, axis=1) * np.linalg.norm(own, axis=1))
        stats[key] = {"feature_sd": te.std(axis=0).mean(), "sd_ratio_to_base": te.std(axis=0).mean() / base_sd,
                      "row_cosine_to_own": cos_own.mean()}

    def run(key, dim):
        tr, te = feats[key]
        r = fit_one("pooler", dim, tr.astype(np.float32), y_tr, te.astype(np.float32), y_te)
        return {"model": key[0], "readout": key[1], "dim": dim, "f1": r["f1"] * 100, **stats[key],
                "converged": r["converged"]}

    rows = Parallel(n_jobs=-1)(delayed(run)(k, d) for k in feats for d in args.dims)
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"pooler_swap_reverse_{args.tag}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(table.pivot_table(index=["model", "readout"], columns="dim", values="f1").round(2).to_string())
        print(table.drop_duplicates(["model", "readout"])[["model", "readout", "feature_sd", "sd_ratio_to_base",
                                                           "row_cosine_to_own"]].round(4).to_string(index=False))
    print(f"all converged: {table['converged'].all()}\nSaved {out}")


if __name__ == "__main__":
    main()
