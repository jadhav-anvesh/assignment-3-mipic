"""Re-evaluate linear probes on existing E3 checkpoints using the expanded grid.

Avoids retraining representation models if checkpoints exist.
Uses candidate_c = (0.01, 0.1, 1.0, 10.0, 100.0) with strict training-fold CV isolation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer
from routing_v2.src.evaluation.evaluate import evaluate_all_tasks
from routing_v2.src.experiments.nested_dimensions import (
    NESTED_DIMS,
    CANONICAL_TASKS,
    compute_effective_dimension,
)
from routing_v2.scripts.prepare_eval_data import download_and_prepare_eval_datasets


def reprobe_e3(
    config_path: str = "routing_v2/configs/e3_nested_dimensions.yaml",
    output_dir: str = "routing_v2/outputs/e3_nested_dimensions",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_tasks_config = download_and_prepare_eval_datasets()

    base_out = Path(output_dir)
    seeds: List[int] = config["training"].get("seeds", [42, 123, 456])
    dev = torch.device(device)

    tokenizer = load_tinybert_tokenizer()
    results_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        seed_dir = base_out / f"seed_{seed}"
        ckpt_path = seed_dir / "final_model.pt"

        if not ckpt_path.exists():
            print(f"[-] Checkpoint not found at {ckpt_path}.")
            print(f"    Please run full training: python routing_v2/scripts/run_experiment.py --config {config_path}")
            return False

        print(f"\n[+] Loading checkpoint for Seed {seed}: {ckpt_path}")
        meta_path = seed_dir / "checkpoint_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
                print(f"    Checkpoint Metadata: model={meta.get('model_name')} (dim={meta.get('embedding_dim')}), updates={meta.get('total_updates')}, lr={meta.get('learning_rate')}")
        model, _ = load_tinybert_6l(device=dev)
        state_dict = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"    Re-running primary probe evaluation with grid C in [0.01, 0.1, 1.0, 10.0, 100.0]...")
        matrix_test, matrix_val, matrix_c = evaluate_all_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks_config=eval_tasks_config,
            dimension_mode="nested",
            nested_dims=NESTED_DIMS,
            device=dev,
            tune_c=True,
            seed=seed,
            return_val_scores=True,
            return_c=True,
            normalize_prefixes=True,
        )

        print(f"    Re-running sensitivity evaluation (fixed C=1.0)...")
        matrix_fixed_c = evaluate_all_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks_config=eval_tasks_config,
            dimension_mode="nested",
            nested_dims=NESTED_DIMS,
            device=dev,
            tune_c=False,
            seed=seed,
            return_val_scores=False,
            normalize_prefixes=True,
        )

        task_analysis = {}
        for task_name in matrix_test.keys():
            val_scores = matrix_val[task_name]
            test_scores = matrix_test[task_name]
            is_reg = (eval_tasks_config.get(task_name, {}).get("type") == "pair_regression")
            analysis = compute_effective_dimension(
                val_scores=val_scores,
                test_scores=test_scores,
                is_regression=is_reg,
                tolerance=float(config.get("experiment", {}).get("tolerance", 0.01)),
            )
            task_analysis[task_name] = analysis

        # Load existing metrics to preserve training stats (loss trajectory, windows, etc.)
        existing_metrics_file = seed_dir / "metrics.json"
        existing_data = {}
        if existing_metrics_file.exists():
            with open(existing_metrics_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

        seed_output = {
            "seed": seed,
            "nested_dims": list(NESTED_DIMS),
            "metrics": matrix_test,
            "val_metrics": matrix_val,
            "metrics_fixed_c": matrix_fixed_c,
            "selected_c": matrix_c,
            "analysis": task_analysis,
            "epoch_average_loss": existing_data.get("epoch_average_loss", 0.0),
            "prefix_losses": existing_data.get("prefix_losses", {}),
            "final_step_dim_losses": existing_data.get("final_step_dim_losses", {}),
            "loss_trajectory": existing_data.get("loss_trajectory", []),
            "final_window_avg_loss": existing_data.get("final_window_avg_loss", 0.0),
            "penultimate_window_avg_loss": existing_data.get("penultimate_window_avg_loss", 0.0),
            "final_loss_drift": existing_data.get("final_loss_drift", 0.0),
        }
        with open(existing_metrics_file, "w", encoding="utf-8") as f:
            json.dump(seed_output, f, indent=2)

        results_by_seed[f"seed_{seed}"] = seed_output

    with open(base_out / "e3_nested_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_by_seed, f, indent=2)

    print(f"\n[✓] All seeds re-evaluated and saved to {base_out} successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-probe E3 models with expanded C grid.")
    parser.add_argument("--config", default="routing_v2/configs/e3_nested_dimensions.yaml")
    parser.add_argument("--output_dir", default="routing_v2/outputs/e3_nested_dimensions")
    args = parser.parse_args()
    reprobe_e3(args.config, args.output_dir)
