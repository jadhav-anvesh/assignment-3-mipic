"""Training-free kNN evaluation of the pooler prefix (Finding 1, check 7).

For each model and width d, every test row is classified by majority vote over
its k nearest probe-train rows (all 26,732 as the database), using the first d
pooler_output coordinates. No classifier is fitted. Metrics:

    cosine        cosine similarity on the raw prefix
    l2norm_euc    Euclidean distance after L2-normalising each prefix. Its
                  neighbour ranking equals cosine's, since ||a-b||^2 = 2 - 2cos
                  for unit vectors, so it is a consistency check.
    raw_euc       Euclidean distance on the raw prefix, the scale-sensitive contrast

Vote ties are broken by the summed similarity of the tied classes' neighbours.

Outputs experiments/layerwise_probe/results/probe_knn_{tag}.csv (model, metric, k, dim, f1, accuracy).

Example, from the repository root:
    python experiments/layerwise_probe/probe_knn.py --models base mrl_s42 mrl_pic_s42 mipic_s42 mrl_sia_s42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.layerwise_probe.probe_size import load_pooler  # noqa: E402

METRICS = ("cosine", "l2norm_euc", "raw_euc")
CHUNK = 256


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def neighbours(x_train, x_test, metric: str, k_max: int):
    """Indices and similarities (larger = closer) of the k_max nearest rows, nearest first."""
    if metric == "cosine":
        db, q = unit(x_train), unit(x_test)
    elif metric == "l2norm_euc":
        db, q = unit(x_train), unit(x_test)
    else:
        db, q = x_train, x_test
    db_sq = (db.astype(np.float64) ** 2).sum(1)
    out_idx = np.empty((len(q), k_max), dtype=np.int64)
    out_sim = np.empty((len(q), k_max))
    for s in range(0, len(q), CHUNK):
        block = q[s:s + CHUNK].astype(np.float64)
        dot = block @ db.T.astype(np.float64)
        if metric == "cosine":
            sim = dot
        else:
            sim = -((block ** 2).sum(1)[:, None] - 2 * dot + db_sq[None, :])  # negative squared distance
        part = np.argpartition(-sim, k_max - 1, axis=1)[:, :k_max]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1, kind="stable")
        out_idx[s:s + CHUNK] = np.take_along_axis(part, order, axis=1)
        out_sim[s:s + CHUNK] = np.take_along_axis(part_sim, order, axis=1)
    return out_idx, out_sim


def vote(nbr_idx, nbr_sim, y_train, k: int, n_classes: int) -> np.ndarray:
    labels = y_train[nbr_idx[:, :k]]
    sims = nbr_sim[:, :k]
    counts = np.zeros((len(labels), n_classes))
    simsum = np.zeros((len(labels), n_classes))
    for c in range(n_classes):
        hit = labels == c
        counts[:, c] = hit.sum(1)
        simsum[:, c] = np.where(hit, sims, 0).sum(1)
    # Rank by count, then by summed similarity (scaled into (0, 1) so it only breaks ties).
    span = simsum.max(1, keepdims=True) - simsum.min(1, keepdims=True) + 1e-12
    return np.argmax(counts + 0.5 * (simsum - simsum.min(1, keepdims=True)) / span, axis=1)


def main(argv: Optional[List[str]] = None) -> None:
    from sklearn.metrics import accuracy_score, f1_score

    parser = argparse.ArgumentParser(description="kNN majority-vote evaluation of pooler prefixes.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 64, 768])
    parser.add_argument("--tag", default="s42")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "features")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results")
    args = parser.parse_args(argv)

    rows = []
    start = time.perf_counter()
    y_ref = None
    for m in args.models:
        x_tr, y_tr, x_te, y_te = load_pooler(args.features_dir, m)
        if y_ref is None:
            y_ref = (y_tr, y_te)
        elif not (np.array_equal(y_tr, y_ref[0]) and np.array_equal(y_te, y_ref[1])):
            raise ValueError(f"{m} has different labels")
        n_classes = int(y_tr.max()) + 1
        for dim in args.dims:
            for metric in METRICS:
                idx, sim = neighbours(x_tr[:, :dim], x_te[:, :dim], metric, max(args.ks))
                for k in args.ks:
                    pred = vote(idx, sim, y_tr, k, n_classes)
                    rows.append({"model": m, "metric": metric, "k": k, "dim": dim,
                                 "f1": f1_score(y_te, pred, average="macro") * 100,
                                 "accuracy": accuracy_score(y_te, pred) * 100})
        print(f"{m} done ({time.perf_counter() - start:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"probe_knn_{args.tag}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.pivot_table(index=["metric", "k", "dim"], columns="model", values="f1")[args.models].round(2).to_string())
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
