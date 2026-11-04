"""STS-B test Spearman for every (layer, width) [CLS] prefix plus pooler_output.

Encodes sentence1 and sentence2 of the STS-B test split with the frozen encoder,
truncates each vector to its first d coordinates, and correlates the cosine
similarity with the gold scores. Rows are layers 1-6 ([CLS] from
hidden_states[layer][:, 0, :]) and "pooler", matching probe_grid.py.

data/final_data.csv (the contrastive training corpus) contains STS-B rows, so
the script reports how many test sentences appear there verbatim and can drop
the affected pairs with --drop-overlap.

Example, from the repository root:
    python h11_probe/sts_eval.py --name base
    python h11_probe/sts_eval.py --name base --drop-overlap
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
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from h11_probe.extract_cls import extract_split  # noqa: E402
from h11_probe.load_encoder import load_encoder  # noqa: E402
from h11_probe.probe_grid import DIMS, PIC_CHECKPOINTS, ROWS  # noqa: E402

STSB_DATASET = "sentence-transformers/stsb"
TRAIN_CORPUS_CSV = REPO_ROOT / "data" / "final_data.csv"


def load_stsb_test() -> pd.DataFrame:
    from datasets import load_dataset

    frame = load_dataset(STSB_DATASET, split="test").to_pandas()
    frame["sentence1"] = frame["sentence1"].astype(str)
    frame["sentence2"] = frame["sentence2"].astype(str)
    return frame


def find_overlap(pairs: pd.DataFrame, out_dir: Path) -> pd.Series:
    """Return a boolean mask of pairs where either sentence is in the training corpus."""
    corpus = pd.read_csv(TRAIN_CORPUS_CSV)
    if "text" not in corpus.columns:
        raise ValueError(f"{TRAIN_CORPUS_CSV} has no 'text' column: {corpus.columns.tolist()}")
    corpus_texts = set(corpus["text"].astype(str))

    in1 = pairs["sentence1"].isin(corpus_texts)
    in2 = pairs["sentence2"].isin(corpus_texts)
    overlapping = sorted(set(pairs.loc[in1, "sentence1"]) | set(pairs.loc[in2, "sentence2"]))
    affected = in1 | in2

    stripped = {t.strip().lower() for t in corpus_texts}
    loose = (pairs["sentence1"].str.strip().str.lower().isin(stripped)
             | pairs["sentence2"].str.strip().str.lower().isin(stripped))

    print(f"Contamination: {int(in1.sum() + in2.sum())} test sentence slots "
          f"({len(overlapping)} unique sentences) appear exactly in {TRAIN_CORPUS_CSV.name}; "
          f"{int(affected.sum())} of {len(pairs)} pairs affected")
    print(f"  (ignoring case and surrounding whitespace: {int(loose.sum())} pairs affected)")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlap_path = out_dir / "sts_overlap.txt"
    overlap_path.write_text("\n".join(overlapping) + "\n", encoding="utf-8")
    print(f"  overlapping sentences saved to {overlap_path}")
    return affected


def prefix_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STS-B Spearman for every (layer, width) prefix.")
    parser.add_argument("--name", required=True, help="Model name for the output file, e.g. base, mipic.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Omit for base TinyBERT.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--drop-overlap", action="store_true",
                        help="Drop pairs where either sentence appears in data/final_data.csv.")
    args = parser.parse_args(argv)

    pairs = load_stsb_test()
    print(f"STS-B test: {len(pairs)} pairs, score range "
          f"{pairs['score'].min():.3f} to {pairs['score'].max():.3f}")

    affected = find_overlap(pairs, args.out_dir)
    if args.drop_overlap:
        pairs = pairs.loc[~affected].reset_index(drop=True)
        print(f"--drop-overlap: kept {len(pairs)} pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer = load_encoder(args.checkpoint, device=device)
    print(f"Device: {device}")
    first = extract_split(encoder, tokenizer, pairs["sentence1"].tolist(),
                          args.batch_size, args.max_length, desc=f"{args.name}/sentence1")
    second = extract_split(encoder, tokenizer, pairs["sentence2"].tolist(),
                           args.batch_size, args.max_length, desc=f"{args.name}/sentence2")
    gold = pairs["score"].to_numpy()

    records = []
    for row in ROWS:
        for dim in DIMS:
            cosine = prefix_cosine(first[row][:, :dim], second[row][:, :dim])
            if not np.isfinite(cosine).all():
                raise ValueError(f"non-finite cosine at layer {row} dim {dim}")
            records.append({
                "model": args.name,
                "layer": row,
                "dim": dim,
                "is_pic_checkpoint": (row, dim) in PIC_CHECKPOINTS,
                "spearman": float(spearmanr(cosine, gold).statistic),
                "n_pairs": len(pairs),
                "dropped_overlap": bool(args.drop_overlap),
            })

    table = pd.DataFrame(records)
    suffix = "_sts_clean" if args.drop_overlap else "_sts"
    out_path = args.out_dir / f"{args.name}{suffix}.csv"
    table.to_csv(out_path, index=False)

    pivot = (table.pivot(index="layer", columns="dim", values="spearman") * 100).reindex(list(ROWS))
    label = "overlap dropped" if args.drop_overlap else "all pairs"
    print(f"\nSTS-B test Spearman x100 for {args.name} ({label}, n={len(pairs)}):")
    print(pivot.round(2).to_string())
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
