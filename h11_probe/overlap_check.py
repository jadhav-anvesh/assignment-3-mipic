"""Overlap of the sentiment probe splits with the training corpus (Finding 1, check 3).

Counts rows of data/tweet_train.csv and data/tweet_test.csv whose text also
appears in data/final_data.csv (the contrastive training corpus), against all
sources and against the TWEET_EVAL_SENT source alone. Two match levels:

    exact        identical strings
    normalised   lower-cased, whitespace collapsed, stripped

Writes h11_probe/results/overlap_check.txt.

Example, from the repository root:
    python h11_probe/overlap_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

import pandas as pd  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def main() -> None:
    corpus = pd.read_csv(DATA / "final_data.csv")
    tweet_src = corpus[corpus["source"] == "TWEET_EVAL_SENT"]
    lines = [f"final_data.csv: {len(corpus)} rows, TWEET_EVAL_SENT: {len(tweet_src)} rows"]
    for split in ("tweet_train", "tweet_test"):
        df = pd.read_csv(DATA / f"{split}.csv")
        for label, pool in (("all sources", corpus), ("TWEET_EVAL_SENT", tweet_src)):
            exact = df["text"].astype(str).isin(set(pool["text"].astype(str)))
            normed = df["text"].map(norm).isin(set(pool["text"].map(norm)))
            lines.append(f"{split} ({len(df)} rows) vs {label}: exact {int(exact.sum())}, normalised {int(normed.sum())}"
                         f" ({normed.mean() * 100:.1f}%)")
            if split == "tweet_train" and label == "TWEET_EVAL_SENT":
                counts = df.loc[exact, "label"].value_counts().sort_index().to_dict()
                lines.append(f"  labels of the exact matches: {counts}")
        # How many corpus tweets are found in this split at all.
        found = tweet_src["text"].astype(str).isin(set(df["text"].astype(str)))
        lines.append(f"  TWEET_EVAL_SENT rows found in {split}: {int(found.sum())} of {len(tweet_src)}")
    text = "\n".join(lines)
    print(text)
    out = REPO_ROOT / "h11_probe" / "results" / "overlap_check.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
