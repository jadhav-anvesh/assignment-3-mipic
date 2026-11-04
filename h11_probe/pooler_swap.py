"""Pooler-weight swap: does the trained pooler alone change the sentiment probe? (Finding 9)

Takes base TinyBERT's saved layer 6 [CLS] features (features/base/*_layer6.npy)
and passes them through each model's pooler, tanh(W h6 + b), with W and b read
from the checkpoint. The encoder output is held at base, so any change in probe
F1 or feature scale is due to the pooler weights alone. The probe is
probe_grid.fit_one, the same default probe as everywhere else.

Features are recomputed in float64 on CPU and cast to float32, so the base row
can differ from base_grid.csv (stored GPU features) by a few hundredths.

Outputs h11_probe/results/pooler_swap_{tag}.csv (pooler_weights, dim, f1,
feature_sd, sd_ratio_to_base, row_cosine_to_base, converged).

Example, from the repository root:
    python h11_probe/pooler_swap.py
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
from joblib import Parallel, delayed  # noqa: E402

from h11_probe.load_encoder import load_encoder  # noqa: E402
from h11_probe.probe_grid import fit_one  # noqa: E402

MODELS = ("mrl", "mrl_pic", "mipic", "mrl_sia")


def pool(encoder, h6: np.ndarray) -> np.ndarray:
    state = encoder.state_dict()
    weight = state["pooler.dense.weight"].double().numpy()
    bias = state["pooler.dense.bias"].double().numpy()
    return np.tanh(h6 @ weight.T + bias)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Probe trained pooler weights on base layer 6 features.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "h11_probe" / "runs" / "seed42" / "checkpoints")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--tag", default="s42")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features" / "base")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "results")
    args = parser.parse_args(argv)

    f = args.features_dir
    h_train = np.load(f / "train_layer6.npy").astype(np.float64)
    h_test = np.load(f / "test_layer6.npy").astype(np.float64)
    y_train, y_test = np.load(f / "train_labels.npy"), np.load(f / "test_labels.npy")
    stored = np.load(f / "test_pooler.npy").astype(np.float64)

    base_encoder, _ = load_encoder(None, device="cpu")
    base_test = pool(base_encoder, h_test)
    print(f"base recompute vs stored test_pooler.npy: max abs diff {np.abs(base_test - stored).max():.2e}")

    feats = {"base": (pool(base_encoder, h_train), base_test)}
    for m in args.models:
        encoder, _ = load_encoder(args.runs_dir / m / "best.pt", device="cpu")
        feats[f"{m}_{args.tag}"] = (pool(encoder, h_train), pool(encoder, h_test))

    base_sd = base_test.std(axis=0).mean()
    stats = {}
    for name, (_, te) in feats.items():
        cos = (te * base_test).sum(1) / (np.linalg.norm(te, axis=1) * np.linalg.norm(base_test, axis=1))
        stats[name] = {"feature_sd": te.std(axis=0).mean(), "sd_ratio_to_base": te.std(axis=0).mean() / base_sd,
                       "row_cosine_to_base": cos.mean()}

    def run(name, dim):
        tr, te = feats[name]
        r = fit_one("pooler", dim, tr.astype(np.float32), y_train, te.astype(np.float32), y_test)
        return {"pooler_weights": name, "h6": "base", "dim": dim, "f1": r["f1"] * 100,
                **stats[name], "converged": r["converged"]}

    rows = Parallel(n_jobs=-1)(delayed(run)(n, d) for n in feats for d in args.dims)
    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"pooler_swap_{args.tag}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 200):
        print(table.pivot(index="pooler_weights", columns="dim", values="f1").round(2).to_string())
        print(pd.DataFrame(stats).T.round(4).to_string())
    print(f"all converged: {table['converged'].all()}\nSaved {out}")


if __name__ == "__main__":
    main()
