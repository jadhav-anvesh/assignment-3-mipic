"""Is base TinyBERT's pooler trained? (Finding 1 check 1, Finding 9)

Compares the pooler weight of huawei-noah/TinyBERT_General_6L_768D with what an
iid Gaussian matrix of the same shape and standard deviation would give, and
with trained weights in the same checkpoint (every encoder linear layer):

    kurtosis              Pearson (not excess); a Gaussian gives 3
    top_sv_ratio          top singular value / sigma * (sqrt(m) + sqrt(n)),
                          the Marchenko-Pastur edge; about 1 for iid noise
    row_norm_cv           CV of row norms; iid Gaussian gives about 1/sqrt(2n)
    bias_max_abs          largest absolute bias entry

It also reads the raw checkpoint for the pretraining heads (cls.*), and
measures how linear the pooler's tanh is on real inputs, using base's saved
test layer 6 [CLS] features: the largest |pre-activation| and the largest
relative gap (|x| - |tanh x|) / |x|.

Outputs h11_probe/results/pooler_check.csv (one row per tensor) and prints the
tanh linearity numbers.

Example, from the repository root:
    python h11_probe/pooler_check.py
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

from h11_probe.load_encoder import TINYBERT_6L_MODEL_NAME, load_encoder  # noqa: E402


def stats(name: str, w: np.ndarray) -> dict:
    w = w.astype(np.float64)
    row = {"tensor": name, "shape": "x".join(map(str, w.shape)), "std": w.std(),
           "kurtosis": float(((w - w.mean()) ** 4).mean() / w.var() ** 2)}
    if w.ndim == 2:
        m, n = w.shape
        row["top_sv_ratio"] = float(np.linalg.svd(w, compute_uv=False)[0] / (w.std() * (np.sqrt(m) + np.sqrt(n))))
        norms = np.linalg.norm(w, axis=1)
        row["row_norm_cv"] = float(norms.std() / norms.mean())
        row["row_norm_cv_iid"] = float(1 / np.sqrt(2 * n))
    else:
        row["max_abs"] = float(np.abs(w).max())
    return row


def raw_checkpoint() -> dict:
    """Every tensor in the hub checkpoint, including the cls.* heads AutoModel drops."""
    from huggingface_hub import hf_hub_download

    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(TINYBERT_6L_MODEL_NAME, filename)
        except Exception:
            continue
        if filename.endswith(".safetensors"):
            from safetensors.numpy import load_file
            return dict(load_file(path))
        import torch
        return {k: v.float().numpy() for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
    raise FileNotFoundError("no checkpoint file found for " + TINYBERT_6L_MODEL_NAME)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Randomness statistics of base TinyBERT's pooler.")
    parser.add_argument("--features-dir", type=Path, default=REPO_ROOT / "h11_probe" / "features" / "base")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "h11_probe" / "results" / "pooler_check.csv")
    args = parser.parse_args(argv)

    encoder, _ = load_encoder(None, device="cpu")
    state = {k: v.detach().double().numpy() for k, v in encoder.state_dict().items() if v.is_floating_point()}
    rows = [stats("pooler.dense.weight", state["pooler.dense.weight"]),
            stats("pooler.dense.bias", state["pooler.dense.bias"])]
    for k, v in state.items():
        if k.startswith("encoder.layer.") and k.endswith(".weight") and v.ndim == 2:
            rows.append(stats(k, v))
    raw = raw_checkpoint()
    for k, v in raw.items():
        if k.startswith("cls.") and k.endswith("bias"):
            rows.append(stats("raw:" + k, np.asarray(v)))

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(table.round(4).to_string(index=False))

    h6 = np.load(args.features_dir / "test_layer6.npy").astype(np.float64)
    pre = h6 @ state["pooler.dense.weight"].T + state["pooler.dense.bias"]
    big = np.abs(pre) > 1e-6
    gap = (np.abs(pre[big]) - np.abs(np.tanh(pre[big]))) / np.abs(pre[big])
    print(f"\ntanh linearity on base test layer 6: max |pre-activation| {np.abs(pre).max():.4f}, "
          f"max relative tanh gap {gap.max() * 100:.2f}%")
    print(f"layer 6 [CLS] mean per-coordinate SD {h6.std(axis=0).mean():.4f}; "
          f"pooler output SD {np.tanh(pre).std(axis=0).mean():.4f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
