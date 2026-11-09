"""Run the SIA/PIC experiment across seeds and datasets.

Training is delegated to ``run_experiments.py`` so its checkpointing and
distributed behavior remain authoritative. This module adds the missing
multi-seed orchestration, Banking77/STS-B evaluation, and seed-level
contrasts/confidence intervals.

Classification CSVs must contain ``text`` and ``label`` columns. STS-B CSVs
must contain ``sentence1``, ``sentence2``, and either ``score`` or ``label``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Keep shared repository modules importable when this file is run directly from
# the experiment folder.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr, t
from torch.utils.data import DataLoader, Dataset

from evaluate import MRLEvaluationConfig, evaluate_tweeteval
from models.mipic import MIPICConfig, TinyBERT_MIPIC
from models.mrl_pic import TinyBERT_MRL_PIC
from models.mrl_sia import MRLSIAConfig, TinyBERT_MRL_SIA
from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer

MODELS = ("mrl", "mrl_sia", "mrl_pic", "mipic")
DIMS = (16, 32, 64, 128, 256, 512, 768)


class STSDataset(Dataset[Tuple[str, str, float]]):
    def __init__(self, path: str | Path) -> None:
        import pandas as pd

        frame = pd.read_csv(path)
        required = {"sentence1", "sentence2"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing STS-B columns: {sorted(missing)}")
        score_column = "score" if "score" in frame.columns else "label" if "label" in frame.columns else None
        if score_column is None:
            raise ValueError(f"{path} must contain an STS-B score or label column.")
        self.samples = [
            (str(a), str(b), float(score))
            for a, b, score in zip(frame["sentence1"], frame["sentence2"], frame[score_column])
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str, float]:
        return self.samples[index]


def _sts_collate(tokenizer: object, max_length: int):
    def collate(batch: Iterable[Tuple[str, str, float]]) -> Dict[str, object]:
        first, second, scores = zip(*batch)
        a = tokenizer(list(first), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        b = tokenizer(list(second), truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        return {"a": a, "b": b, "scores": torch.tensor(scores, dtype=torch.float32)}

    return collate


def evaluate_stsb(
    model: torch.nn.Module,
    tokenizer: object,
    path: str | Path,
    batch_size: int = 64,
    max_length: int = 256,
) -> Dict[int, float]:
    """Return Spearman x100 for every nested width on an STS-B split."""
    model.eval()
    loader = DataLoader(
        STSDataset(path),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_sts_collate(tokenizer, max_length),
    )
    device = next(model.parameters()).device
    embeddings_a: List[np.ndarray] = []
    embeddings_b: List[np.ndarray] = []
    scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            output_a = model(
                input_ids=batch["a"]["input_ids"].to(device),
                attention_mask=batch["a"]["attention_mask"].to(device),
                output_hidden_states=True,
                return_dict=True,
            )
            output_b = model(
                input_ids=batch["b"]["input_ids"].to(device),
                attention_mask=batch["b"]["attention_mask"].to(device),
                output_hidden_states=True,
                return_dict=True,
            )
            from models.tinybert import mrl_sentence_embedding

            embeddings_a.append(mrl_sentence_embedding(output_a).cpu().numpy())
            embeddings_b.append(mrl_sentence_embedding(output_b).cpu().numpy())
            scores.extend(batch["scores"].tolist())
    first = np.concatenate(embeddings_a)
    second = np.concatenate(embeddings_b)
    target = np.asarray(scores)
    return {
        dimension: float(spearmanr(
            np.sum(first[:, :dimension] * second[:, :dimension], axis=1)
            / (np.linalg.norm(first[:, :dimension], axis=1)
               * np.linalg.norm(second[:, :dimension], axis=1)),
            target,
        ).statistic * 100)
        for dimension in DIMS
    }


def _load_encoder(model_name: str, checkpoint: Path, device: torch.device, kl: bool) -> torch.nn.Module:
    if model_name == "mrl":
        model, _ = load_tinybert_6l(device=device)
    elif model_name == "mrl_sia":
        model = TinyBERT_MRL_SIA(device=device, config=MRLSIAConfig(use_attention_kl=kl))
    elif model_name == "mrl_pic":
        model = TinyBERT_MRL_PIC(device=device)
    elif model_name == "mipic":
        model = TinyBERT_MIPIC(device=device, config=MIPICConfig(use_attention_kl=kl))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    encoder = model if model_name == "mrl" else model.encoder
    encoder.eval()
    return encoder


def _run_training(
    model: str,
    seed: int,
    kl: bool,
    output_dir: Path,
    corpus: Path,
    tweet_train: Path,
    tweet_test: Path,
    fresh: bool,
) -> None:
    env = os.environ.copy()
    run_mode = env.get("MIPIC_RUN_MODE", "single").lower()
    if run_mode not in {"single", "ddp"}:
        raise ValueError(f"MIPIC_RUN_MODE must be 'single' or 'ddp', got {run_mode!r}")
    env.update({
        "MIPIC_SEED": str(seed),
        "MIPIC_OUTPUT_DIR": str(output_dir),
        "MIPIC_TRAIN_CORPUS": str(corpus),
        "MIPIC_EVAL_TRAIN": str(tweet_train),
        "MIPIC_EVAL_TEST": str(tweet_test),
        "MIPIC_ATTENTION_KL": "1" if kl else "0",
        "MIPIC_FRESH": "1" if fresh else "0",
        "MIPIC_SKIP_EVALUATION": "1",
    })
    if run_mode == "ddp":
        env.update({
            "MIPIC_BATCH_SIZE": "8",
            "MIPIC_DDP_STATIC_GRAPH": "1",
            "MIPIC_DDP_TIMEOUT_MIN": "20",
            "NCCL_P2P_DISABLE": "1",
        })
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            str(REPO_ROOT / "run_experiments.py"),
        ]
    else:
        command = [sys.executable, str(REPO_ROOT / "run_experiments.py")]
    subprocess.run(
        command + ["--model", model],
        check=True,
        env=env,
    )


def _mean_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    if len(values) < 2:
        raise ValueError("At least two seed values are required for a confidence interval.")
    mean = float(np.mean(values))
    critical = float(t.ppf(0.975, len(values) - 1))
    half_width = critical * float(np.std(values, ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def _contrasts(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    grouped: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["kl"]),
            int(row["dimension"]),
            int(row["seed"]),
        )
        grouped.setdefault(key, {})[str(row["model"])] = float(row["score"])
    seed_contrasts: Dict[Tuple[str, str, int, str], List[float]] = {}
    for (dataset, kl, dimension, _seed), values in sorted(grouped.items()):
        if set(MODELS).difference(values):
            continue
        mrl, sia, pic, mipic = (values[name] for name in MODELS)
        contrasts = {
            "sia_effect": mipic - pic,
            "interaction": mipic - sia - pic + mrl,
        }
        if dimension == 768:
            contrasts["full_width_gap"] = mipic - mrl
        for quantity, value in contrasts.items():
            seed_contrasts.setdefault((dataset, kl, dimension, quantity), []).append(value)
    out: List[Dict[str, object]] = []
    for (dataset, kl, dimension, quantity), values in sorted(seed_contrasts.items()):
        mean, ci_low, ci_high = _mean_ci(values)
        out.append({
            "dataset": dataset,
            "kl": kl,
            "dimension": dimension,
            "quantity": quantity,
            "seed_values": json.dumps(values),
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            list(out[0])
            if out
            else ["dataset", "kl", "dimension", "quantity", "mean", "ci_low", "ci_high", "n_seeds"]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in out:
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/final_data.csv"))
    parser.add_argument("--tweet-train", type=Path, default=Path("data/tweet_train.csv"))
    parser.add_argument("--tweet-test", type=Path, default=Path("data/tweet_test.csv"))
    parser.add_argument("--banking-train", type=Path)
    parser.add_argument("--banking-test", type=Path)
    parser.add_argument("--stsb-test", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled_experiment"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--kl-on-only",
        action="store_true",
        help="Train and evaluate only the new SIA/MIPIC KL-on arms.",
    )
    parser.add_argument(
        "--kl-off-only",
        action="store_true",
        help="Evaluate only existing KL-off checkpoints from --kl-off-dir.",
    )
    parser.add_argument(
        "--kl-off-dir",
        type=Path,
        help="Directory containing seedN/checkpoints/<model>/best.pt KL-off checkpoints.",
    )
    parser.add_argument("--fresh", action="store_true", help="Discard existing checkpoints before training.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        raise FileNotFoundError(f"Training corpus not found: {args.corpus}")
    if args.kl_on_only and args.kl_off_only:
        raise ValueError("--kl-on-only and --kl-off-only cannot be combined.")
    if args.kl_off_only and args.kl_off_dir is None:
        raise ValueError("--kl-off-only requires --kl-off-dir.")
    if args.kl_off_dir is not None and not args.kl_off_dir.exists():
        raise FileNotFoundError(f"KL-off checkpoint directory not found: {args.kl_off_dir}")
    datasets = {"tweet": (args.tweet_train, args.tweet_test)}
    if (args.banking_train is None) != (args.banking_test is None):
        raise ValueError("Provide both --banking-train and --banking-test.")
    if args.banking_train is not None:
        datasets["banking77"] = (args.banking_train, args.banking_test)
    for name, paths in datasets.items():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"{name} data not found: {path}")
    if args.stsb_test is not None and not args.stsb_test.exists():
        raise FileNotFoundError(f"STS-B data not found: {args.stsb_test}")

    rows: List[Dict[str, object]] = []
    device = torch.device(args.device)
    tokenizer = load_tinybert_tokenizer()
    evaluation_cache: Dict[Tuple[str, str], Dict[int, Dict[str, float]]] = {}
    sts_cache: Dict[Tuple[str, str], Dict[int, float]] = {}
    kl_values = (True,) if args.kl_on_only else (False,) if args.kl_off_only else (False, True)
    model_values = ("mrl_sia", "mipic") if args.kl_on_only else MODELS
    for seed in args.seeds:
        for kl in kl_values:
            for model_name in model_values:
                run_dir = args.output_dir / f"seed{seed}" / ("kl_on" if kl else "kl_off")
                checkpoint_dir = (
                    args.kl_off_dir / f"seed{seed}"
                    if args.kl_off_only
                    else run_dir
                )
                if not args.kl_on_only and kl and model_name in {"mrl", "mrl_pic"}:
                    checkpoint_dir = args.output_dir / f"seed{seed}" / "kl_off"
                completed_checkpoint = checkpoint_dir / "checkpoints" / model_name / "model.pt"
                if not args.skip_training:
                    train_condition = args.kl_on_only and kl and model_name in {"mrl_sia", "mipic"}
                    if train_condition and (args.fresh or not completed_checkpoint.exists()):
                        _run_training(
                            model_name,
                            seed,
                            kl,
                            run_dir,
                            args.corpus,
                            args.tweet_train,
                            args.tweet_test,
                            args.fresh,
                        )
                checkpoint = checkpoint_dir / "checkpoints" / model_name / "best.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing completed checkpoint: {checkpoint}")
                encoder = _load_encoder(model_name, checkpoint, device, kl)
                config = MRLEvaluationConfig(dimensions=DIMS, batch_size=64, max_length=256)
                for dataset_name, (train_path, test_path) in datasets.items():
                    cache_key = (str(checkpoint.resolve()), dataset_name)
                    if cache_key not in evaluation_cache:
                        evaluation_cache[cache_key] = evaluate_tweeteval(
                            encoder, tokenizer, train_path, test_path, config
                        )
                    metrics = evaluation_cache[cache_key]
                    for dimension, metric in metrics.items():
                        rows.append({
                            "seed": seed, "kl": "on" if kl else "off", "model": model_name,
                            "dataset": dataset_name, "dimension": dimension, "score": metric["f1"] * 100,
                        })
                if args.stsb_test is not None:
                    sts_key = (str(checkpoint.resolve()), str(args.stsb_test.resolve()))
                    if sts_key not in sts_cache:
                        sts_cache[sts_key] = evaluate_stsb(encoder, tokenizer, args.stsb_test)
                    for dimension, score in sts_cache[sts_key].items():
                        rows.append({
                            "seed": seed, "kl": "on" if kl else "off", "model": model_name,
                            "dataset": "stsb", "dimension": dimension, "score": score,
                        })
                del encoder
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    _contrasts(rows, args.output_dir / "contrasts.csv")
    print(f"Saved scores and contrasts under {args.output_dir}")


if __name__ == "__main__":
    main()
