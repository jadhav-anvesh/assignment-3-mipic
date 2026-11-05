"""Train mrl_sia or mipic with the SIA attention-KL term switched ON.

The code used for the h11_probe runs hardcoded ``use_attention_kl=False`` in
models/mrl_sia.py and models/mrl_sia_pic.py (now the MIPIC_ATTENTION_KL switch,
default on; this wrapper forces KL on whatever the switch says), while the README's Kaggle run
(code version MIPIC_FINAL_V20) logged a nonzero ``sia_attn_loss``. This wrapper
reproduces the KL-on configuration without editing any repository file:

* It subclasses TinyBERT_MRL_SIA / TinyBERT_MIPIC, sets
  ``use_attention_kl = True`` on every SIALoss inside the model right after
  construction, and swaps the subclass into run_experiments' namespace.
* It then calls the unmodified ``run_experiments.run_experiment(model)``, so
  data, optimizer, scheduler, AMP, epoch callbacks, held-out best-epoch
  selection, best.pt / model.pt and the TweetEval evaluation are identical.
* Training writes to a staging folder. Afterwards results are renamed to
  ``mrl_sia_kl`` / ``mipic_kl`` (also ``*_kl_last_epoch``) and the checkpoints
  are moved to ``<output-dir>/checkpoints/<model>_kl/``, so nothing can be
  confused with a KL-off run.

Guard: if ``sia_attn_loss`` is exactly 0 on every one of the first 50 training
steps, training stops with an error, because that means KL did not turn on.

Examples, from the repository root:
    python h11_probe/train_kl_on.py --model mrl_sia --seed 42 --output-dir h11_probe/runs_kl/seed42
    python h11_probe/train_kl_on.py --model mipic --seed 42 --output-dir h11_probe/runs_kl/seed42 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

KL_CHECK_STEPS = 50
KL_OFF_NAMES = ("mrl", "mrl_sia", "mrl_pic", "mipic")
SMOKE_VARS = ("MIPIC_MAX_BATCHES", "MIPIC_SMOKE_FULL_PIPELINE", "MIPIC_EVAL_TRAIN_SAMPLES", "MIPIC_EVAL_TEST_SAMPLES")


class KLNotActiveError(RuntimeError):
    pass


def sia_modules(model) -> list:
    from losses.sia_loss import SIALoss

    return [(name, module) for name, module in model.named_modules() if isinstance(module, SIALoss)]


def make_kl_class(base):
    """Subclass ``base`` so every SIALoss has use_attention_kl=True and KL is checked."""

    class KLOn(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            found = sia_modules(self)
            if not found:
                raise KLNotActiveError(f"{base.__name__} contains no SIALoss module")
            for _, module in found:
                module.use_attention_kl = True
            # The module the training loss actually calls must be one of them.
            if not self.sia_loss_fn.use_attention_kl:
                raise KLNotActiveError("sia_loss_fn.use_attention_kl is still False after setting it")
            self.kl_steps_checked = 0
            self.kl_steps_nonzero = 0

        def compute_losses(self, *args, **kwargs):
            out = super().compute_losses(*args, **kwargs)
            if self.training and self.kl_steps_checked < KL_CHECK_STEPS:
                self.kl_steps_checked += 1
                if out["sia_attn_loss"].detach().float().item() != 0.0:
                    self.kl_steps_nonzero += 1
                if self.kl_steps_checked == KL_CHECK_STEPS:
                    assert_kl_active(self)
            return out

    KLOn.__name__ = f"{base.__name__}_KLOn"
    KLOn.__qualname__ = KLOn.__name__
    return KLOn


def assert_kl_active(model) -> None:
    if model.kl_steps_checked == 0:
        raise KLNotActiveError("No training steps went through compute_losses; KL could not be checked")
    if model.kl_steps_nonzero == 0:
        raise KLNotActiveError(
            f"sia_attn_loss was exactly 0 on all of the first {model.kl_steps_checked} training steps: "
            "the attention KL term is NOT active. Stopping."
        )
    print(f"KL check passed: sia_attn_loss nonzero on {model.kl_steps_nonzero} of the first "
          f"{model.kl_steps_checked} training steps.", flush=True)


def rename_results(staged: Dict, model: str, new_name: str) -> Dict[str, Dict]:
    """Pull the entries for ``model`` out of a staged results JSON, under ``new_name``."""
    out: Dict[str, Dict] = {}
    for section in ("training_history", "evaluations", "epoch_selection", "evaluation_errors"):
        entries = staged.get(section, {})
        for suffix in ("", "_last_epoch"):
            if model + suffix in entries:
                out.setdefault(section, {})[new_name + suffix] = entries[model + suffix]
    return out


def repo_relative(path: Path) -> str:
    """Path relative to the repository root when it lies inside it, so results don't record machine paths."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def merge_into_output(output_dir: Path, staging_dir: Path, model: str, new_name: str, kl_stats: Dict) -> Path:
    with open(staging_dir / "tweeteval_results.json", "r", encoding="utf-8") as handle:
        staged = json.load(handle)
    if model not in staged.get("evaluations", {}):
        raise RuntimeError(
            f"Training finished but {model} has no evaluation entry in {staging_dir}; "
            f"evaluation_errors={staged.get('evaluation_errors')}. Staging folder kept for diagnosis."
        )

    results_path = output_dir / "tweeteval_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {"metadata": {}, "training_history": {}, "evaluations": {}}

    metadata = dict(staged.get("metadata", {}))
    metadata.update({"use_attention_kl": True, "model_name": new_name, "source_arm": model,
                     "output_dir": repo_relative(output_dir), "kl_check": kl_stats})
    results["metadata"] = metadata
    results.setdefault("metadata_by_model", {})[new_name] = metadata
    for section, entries in rename_results(staged, model, new_name).items():
        results.setdefault(section, {}).update(entries)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    import pandas as pd

    rows = [
        {"Model": name, "Dimension": int(dim), "Accuracy": m["accuracy"], "Macro_F1": m["f1"]}
        for name, dims in results.get("evaluations", {}).items()
        for dim, m in dims.items()
    ]
    pd.DataFrame(rows).sort_values(by=["Model", "Dimension"]).to_csv(
        output_dir / "tweeteval_results.csv", index=False
    )

    target = output_dir / "checkpoints" / new_name
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir / "checkpoints" / model), str(target))
    shutil.rmtree(staging_dir)
    return results_path


def refuse_kl_off_folder(output_dir: Path) -> None:
    """Never write KL-on results into a folder that holds KL-off results."""
    results_path = output_dir / "tweeteval_results.json"
    if not results_path.exists():
        return
    with open(results_path, "r", encoding="utf-8") as handle:
        existing = json.load(handle)
    clash = [name for name in existing.get("evaluations", {}) if name.split("_last_epoch")[0] in KL_OFF_NAMES]
    if clash:
        raise SystemExit(f"{results_path} already holds KL-off results ({', '.join(clash)}). "
                         "Use a separate --output-dir, e.g. h11_probe/runs_kl/seed<N>.")


def dry_run(model: str) -> None:
    """Build the KL-on model on CPU, confirm the flag, and run one loss forward pass."""
    import torch
    from models.mipic import TinyBERT_MIPIC
    from models.mrl_sia import TinyBERT_MRL_SIA
    from models.tinybert import load_tinybert_tokenizer
    from train import PairTokenizerCollate, TextPairRaw

    base = {"mrl_sia": TinyBERT_MRL_SIA, "mipic": TinyBERT_MIPIC}[model]
    plain = base(device=torch.device("cpu"))
    print("KL-off model as built by run_experiments.py:")
    for name, module in sia_modules(plain):
        print(f"  {name}: use_attention_kl={module.use_attention_kl}")
    del plain

    kl_model = make_kl_class(base)(device=torch.device("cpu"))
    print(f"KL-on model ({type(kl_model).__name__}):")
    for name, module in sia_modules(kl_model):
        print(f"  {name}: use_attention_kl={module.use_attention_kl}")
    print(f"  sia_loss_fn is that module: {any(m is kl_model.sia_loss_fn for _, m in sia_modules(kl_model))}")

    tokenizer = load_tinybert_tokenizer()
    texts = ["the movie was a pleasant surprise", "my flight got cancelled again and nobody called",
             "how do i reset my card pin", "two dogs are playing in the snow"]
    batch = PairTokenizerCollate(tokenizer, 64)([TextPairRaw(texts)[i] for i in range(len(texts))])
    batch = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    kl_model.train()
    torch.manual_seed(0)
    out = kl_model.compute_losses(batch=batch)
    shown = ", ".join(f"{k}={v.item():.6f}" for k, v in out.items() if torch.is_tensor(v) and v.dim() == 0)
    print(f"One CPU training-mode forward: {shown}")
    print(f"Guard counters after 1 step: checked={kl_model.kl_steps_checked}, nonzero={kl_model.kl_steps_nonzero}")

    # Exercise the guard: switch KL back off and pretend 49 zero-KL steps went by.
    for _, module in sia_modules(kl_model):
        module.use_attention_kl = False
    kl_model.kl_steps_checked, kl_model.kl_steps_nonzero = KL_CHECK_STEPS - 1, 0
    try:
        kl_model.compute_losses(batch=batch)
    except KLNotActiveError as exc:
        print(f"Guard fires when KL is off, as intended: {exc}")
    else:
        raise SystemExit("Guard did NOT fire on a KL-off model")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train mrl_sia or mipic with attention KL on.")
    parser.add_argument("--model", required=True, choices=["mrl_sia", "mipic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU only: build the model, confirm the flag and the guard, do not train.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir)
    output_dir = output_dir.resolve()
    new_name = f"{args.model}_kl"
    staging_dir = output_dir / "_staging" / args.model

    os.chdir(REPO_ROOT)  # run_experiments resolves data/ relative to the working directory
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # keep the dry run off the GPU
        print(f"DRY RUN: would train {args.model} with KL on, seed {args.seed}")
        print(f"  staging: {staging_dir}")
        print(f"  results: {output_dir / 'tweeteval_results.json'} as {new_name}, {new_name}_last_epoch")
        print(f"  checkpoints: {output_dir / 'checkpoints' / new_name}")
        dry_run(args.model)
        return

    refuse_kl_off_folder(output_dir)
    for name in SMOKE_VARS:
        if os.environ.get(name):
            print(f"WARNING: {name}={os.environ[name]} is set; this is a truncated or smoke run.", flush=True)

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    # run_experiments reads these at import time. The working directory is REPO_ROOT here,
    # so the repo-relative form points at the same folder.
    os.environ["MIPIC_OUTPUT_DIR"] = repo_relative(staging_dir)
    os.environ["MIPIC_SEED"] = str(args.seed)
    os.environ["MIPIC_FRESH"] = "1"

    import run_experiments as rx

    base = {"mrl_sia": rx.TinyBERT_MRL_SIA, "mipic": rx.TinyBERT_MIPIC}[args.model]
    kl_class = make_kl_class(base)
    built: List[object] = []

    class Recording(kl_class):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            built.append(self)

    Recording.__name__ = kl_class.__name__
    setattr(rx, base.__name__, Recording)

    print(f"Training {args.model} with use_attention_kl=True; results will be named {new_name}.", flush=True)
    try:
        rx.run_experiment(args.model)
    finally:
        rx.cleanup_distributed()

    if not built:
        raise KLNotActiveError("run_experiments never constructed the KL-on model")
    model = built[-1]
    if model.kl_steps_checked < KL_CHECK_STEPS:
        assert_kl_active(model)  # shorter than 50 steps (smoke run): check what there was
    kl_stats = {"steps_checked": model.kl_steps_checked, "steps_nonzero": model.kl_steps_nonzero}

    results_path = merge_into_output(output_dir, staging_dir, args.model, new_name, kl_stats)
    print(f"\nKL-on results saved as {new_name} in {results_path}")
    print(f"Checkpoints moved to {output_dir / 'checkpoints' / new_name}")


if __name__ == "__main__":
    main()
