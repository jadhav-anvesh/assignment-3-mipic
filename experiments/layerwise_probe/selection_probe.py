"""Epoch-selection probe score of the untrained backbone (Finding 1, check 8).

Calls evaluate.validation_probe_score exactly as run_experiments.py's epoch
selector does: the same tweet_train.csv, and an MRLEvaluationConfig with
dimensions=MRL_NESTED_DIMS, batch_size=64, max_length=256, train_samples=None,
random_state=42, max_iter=1000 (the run_experiments.py values when no MIPIC_*
eval variable is set, as in the seed 42 runs). holdout and max_rows keep their
defaults (0.2 and 6,000), as in run_experiments.py.

Scores base, and each --checkpoint given as name=path, so a best.pt whose epoch
is known can be compared with its logged score as a reproduction check.

Example, from the repository root:
    python experiments/layerwise_probe/selection_probe.py --checkpoint mrl_s42=experiments/layerwise_probe/runs/seed42/checkpoints/mrl/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from evaluate import MRLEvaluationConfig, validation_probe_score  # noqa: E402
from experiments.layerwise_probe.load_encoder import load_encoder  # noqa: E402
from losses.mrl_loss import MRL_NESTED_DIMS  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="validation_probe_score for base and optional checkpoints.")
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path to a trained checkpoint")
    parser.add_argument("--train-csv", type=Path, default=REPO_ROOT / "data" / "tweet_train.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results" / "selection_probe_s42.csv")
    args = parser.parse_args(argv)

    config = MRLEvaluationConfig(
        dimensions=MRL_NESTED_DIMS, batch_size=64, max_length=256,
        train_samples=None, random_state=42, max_iter=1000,
    )
    targets = [("base", None)] + [tuple(c.split("=", 1)) for c in args.checkpoint]
    rows = []
    for name, path in targets:
        encoder, tokenizer = load_encoder(path)
        score = validation_probe_score(encoder, tokenizer, args.train_csv, config) * 100
        print(f"{name}: held-out probe macro-F1 {score:.2f}", flush=True)
        rows.append({"model": name, "checkpoint": path or "", "val_macro_f1": round(score, 3)})
        del encoder

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
