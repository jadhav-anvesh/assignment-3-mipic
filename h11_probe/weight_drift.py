"""How far training moved each part of the encoder from base (Finding 9).

Loads base TinyBERT-6L and each trained checkpoint through load_encoder (CPU,
strict load) and compares their parameters. For a group of tensors G:

    relative change = ||W_trained - W_base||_F / ||W_base||_F   over all of G
    cosine          = <W_trained, W_base> / (||W_trained|| ||W_base||)

Groups: embeddings, each transformer layer 1..6, the pooler, all encoder
parameters except the pooler ("encoder"), and everything ("all"). Biases that
are exactly zero in base (the pooler bias) count in the norms like any other
tensor; the relative change of the pooler is dominated by its weight.

Outputs h11_probe/results/weight_drift_{tag}.csv.

Example, from the repository root:
    python h11_probe/weight_drift.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from h11_probe.load_encoder import load_encoder  # noqa: E402

MODELS = ("mrl", "mrl_pic", "mipic", "mrl_sia")


def group_of(name: str) -> str:
    if name.startswith("embeddings."):
        return "embeddings"
    if name.startswith("encoder.layer."):
        return f"layer{int(name.split('.')[2]) + 1}"
    if name.startswith("pooler."):
        return "pooler"
    return "other"


def drift(base: dict, trained: dict, names: List[str]) -> dict:
    diff_sq = base_sq = trained_sq = dot = 0.0
    for n in names:
        b, t = base[n].double(), trained[n].double()
        diff_sq += float(((t - b) ** 2).sum())
        base_sq += float((b ** 2).sum())
        trained_sq += float((t ** 2).sum())
        dot += float((t * b).sum())
    return {
        "relative_change": (diff_sq ** 0.5) / (base_sq ** 0.5),
        "cosine": dot / ((base_sq * trained_sq) ** 0.5),
        "norm_ratio": (trained_sq / base_sq) ** 0.5,
        "n_params": int(sum(base[n].numel() for n in names)),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Relative parameter change of trained encoders from base.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "h11_probe" / "runs" / "seed42" / "checkpoints")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--tag", default="s42")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "results")
    args = parser.parse_args(argv)

    base_model, _ = load_encoder(None, device="cpu")
    base = {k: v.detach().clone() for k, v in base_model.state_dict().items() if v.is_floating_point()}
    del base_model
    groups = {}
    for n in base:
        groups.setdefault(group_of(n), []).append(n)
    if groups.get("other"):
        raise ValueError(f"unexpected parameter names: {groups['other'][:5]}")
    groups["encoder"] = [n for n in base if group_of(n) != "pooler"]
    groups["all"] = list(base)
    order = ["embeddings"] + [f"layer{i}" for i in range(1, 7)] + ["pooler", "encoder", "all"]

    rows = []
    for m in args.models:
        path = args.runs_dir / m / args.checkpoint_name
        model, _ = load_encoder(path, device="cpu")
        trained = {k: v.detach() for k, v in model.state_dict().items() if k in base}
        for g in order:
            rows.append({"model": f"{m}_{args.tag}", "checkpoint": str(path.relative_to(REPO_ROOT)), "group": g,
                         **drift(base, trained, groups[g])})
        del model

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"weight_drift_{args.tag}.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for col in ("relative_change", "cosine", "norm_ratio"):
            print(f"\n{col}:")
            print(table.pivot(index="group", columns="model", values=col).reindex(order).round(4).to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
