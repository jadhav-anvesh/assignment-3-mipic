"""Check that dynamic padding does not change the saved [CLS] vectors.

Recomputes layer 6 [CLS] for the first few test sentences one at a time (no
batching, no padding) and compares them with the batched float16 values saved by
extract_cls.py. The checkpoint and max_length are read from the feature folder's
meta.json, so the check always uses the encoder that produced the features.

Example, from the repository root:
    python h11_probe/check_padding.py --name base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from h11_probe.load_encoder import load_encoder  # noqa: E402

TEST_CSV = REPO_ROOT / "data" / "tweet_test.csv"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Check that padding does not change saved [CLS] vectors.")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base, mrl, mrl_pic, mipic.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features")
    parser.add_argument("--rows", type=int, default=5, help="Number of test sentences to check.")
    parser.add_argument("--layer", type=int, default=6)
    args = parser.parse_args(argv)

    feature_dir = args.out_dir / args.name
    with open(feature_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    checkpoint = Path(meta["checkpoint"]) if meta["checkpoint"] else None
    max_length = int(meta["max_length"])

    encoder, tokenizer = load_encoder(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(encoder.parameters()).device
    saved = np.load(feature_dir / f"test_layer{args.layer}.npy").astype(np.float32)
    texts = pd.read_csv(TEST_CSV)["text"].astype(str).tolist()[: args.rows]

    worst = 0.0
    failures = 0
    for i, text in enumerate(texts):
        enc = tokenizer([text], truncation=True, padding=False, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            cls = encoder(**enc, output_hidden_states=True).hidden_states[args.layer][0, 0, :].float().cpu().numpy()
        diff = np.abs(cls - saved[i])
        rel = diff.max() / np.abs(cls).max()
        ok = np.allclose(cls, saved[i], rtol=1e-3, atol=1e-2)
        failures += not ok
        worst = max(worst, float(diff.max()))
        print(f"row {i}: tokens={enc['input_ids'].shape[1]:3d}  max|x|={np.abs(cls).max():6.3f}  "
              f"max abs diff={diff.max():.5f}  rel={rel:.2e}  allclose={ok}")

    # A float16 value near |x| has spacing about |x| * 2**-10; half of that is the rounding error.
    print(f"worst abs diff {worst:.5f}; float16 half-spacing at max |x| = "
          f"{np.abs(saved[: args.rows]).max() * 2**-11:.5f}")
    if failures:
        raise SystemExit(f"{failures} of {len(texts)} rows differ beyond float16 tolerance")


if __name__ == "__main__":
    main()
