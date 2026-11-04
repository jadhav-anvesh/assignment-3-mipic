"""Extract per-layer [CLS] vectors for the H11 early-layer probe.

For layers 1-6 this saves hidden_states[layer][:, 0, :] (all 768 coordinates,
float32) for the tweet sentiment train and test splits, so prefixes of any width
can be probed later without re-running the encoder. hidden_states[0] is the
embedding output, so layer L is hidden_states[L]. It also saves pooler_output,
the vector evaluate.py probes, as {split}_pooler.npy.

Example, from the repository root:
    python h11_probe/extract_cls.py --name base --limit 200
    python h11_probe/extract_cls.py --name mipic --checkpoint "results/checkpoints/mipic/best.pt"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from h11_probe.load_encoder import load_encoder  # noqa: E402

LAYERS = (1, 2, 3, 4, 5, 6)
FEATURE_KEYS: Tuple[Union[int, str], ...] = LAYERS + ("pooler",)
SPLITS = {
    "train": REPO_ROOT / "data" / "tweet_train.csv",
    "test": REPO_ROOT / "data" / "tweet_test.csv",
}


def read_split(csv_path: Path, limit: Optional[int]) -> tuple[List[str], np.ndarray]:
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain 'text' and 'label' columns.")
    if limit is not None:
        frame = frame.head(limit)
    # Same casting as evaluate.ClassificationDataset.
    return frame["text"].astype(str).tolist(), frame["label"].astype(int).to_numpy()


def extract_split(
    encoder: torch.nn.Module,
    tokenizer: object,
    texts: List[str],
    batch_size: int,
    max_length: int,
    desc: str,
) -> Dict[Union[int, str], np.ndarray]:
    """Return {layer: [CLS] matrix} for layers 1-6 plus {"pooler": pooler_output}."""
    device = next(encoder.parameters()).device
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: Dict[Union[int, str], List[np.ndarray]] = {key: [] for key in FEATURE_KEYS}
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch"):
            encoded = tokenizer(
                list(batch),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = encoder(**encoded, output_hidden_states=True, return_dict=True)
            for layer in LAYERS:
                chunks[layer].append(output.hidden_states[layer][:, 0, :].float().cpu().numpy())
            # The Linear+Tanh head that evaluate.py and the README numbers read.
            chunks["pooler"].append(output.pooler_output.float().cpu().numpy())
    return {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}


def feature_filename(split: str, key: Union[int, str]) -> str:
    return f"{split}_pooler.npy" if key == "pooler" else f"{split}_layer{key}.npy"


def to_checked_float32(array: np.ndarray, label: str) -> np.ndarray:
    # float32, not float16: evaluate.py probes float32 features, and float16
    # rounding alone moved the base pooler probe by up to 0.19 macro-F1.
    single = array.astype(np.float32)
    bad = ~np.isfinite(single)
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} non-finite values in float32 features")
    return single


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract per-layer [CLS] features for H11.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Omit for base TinyBERT.")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base, mrl, mrl_pic, mipic.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N rows of each split.")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tokenizer = load_encoder(args.checkpoint, device=device)
    print(f"Device: {device}")

    target = args.out_dir / args.name
    target.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": args.name,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "layers": list(LAYERS),
        "pooler": True,
        "dtype": "float32",
        "device": device,
        "date": datetime.now().isoformat(timespec="seconds"),
        "rows": {},
        "seconds": {},
    }

    for split, csv_path in SPLITS.items():
        texts, labels = read_split(csv_path, args.limit)
        start = time.perf_counter()
        features = extract_split(
            encoder, tokenizer, texts, args.batch_size, args.max_length, desc=f"{args.name}/{split}"
        )
        elapsed = time.perf_counter() - start

        # Check everything before writing anything for this split.
        checked = {
            key: to_checked_float32(array, f"{split} {key if key == 'pooler' else f'layer {key}'}")
            for key, array in features.items()
        }
        for key, array in checked.items():
            np.save(target / feature_filename(split, key), array)
        np.save(target / f"{split}_labels.npy", labels)

        meta["rows"][split] = len(labels)
        meta["seconds"][split] = round(elapsed, 2)
        print(f"{split}: {len(labels)} rows in {elapsed:.1f}s ({len(labels) / elapsed:.0f} rows/s)")

    with open(target / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"Saved features to {target}")


if __name__ == "__main__":
    main()
