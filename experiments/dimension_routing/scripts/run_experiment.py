"""Unified experiment execution CLI for V2 routing."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from routing_v2.src.utils.config import load_config
from routing_v2.src.experiments.objective_comparison import run_objective_comparison
from routing_v2.src.experiments.assignment_sweep import run_assignment_sweep
from routing_v2.src.experiments.nested_dimensions import run_nested_experiment
from routing_v2.src.experiments.gradient_analysis import run_gradient_analysis
from routing_v2.scripts.prepare_eval_data import download_and_prepare_eval_datasets


def main():
    parser = argparse.ArgumentParser(description="Run MRL V2 Experiments")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration.")
    parser.add_argument("--output_dir", default=None, help="Optional override for output directory.")
    parser.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--quick_eval", action="store_true", help="Evaluate only on TWEET_EVAL_SENT for fast sanity checks.")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config["experiment"]["name"]
    out_dir = Path(args.output_dir or f"routing_v2/outputs/{exp_name}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {exp_name} on device: {device}")

    if args.quick_eval:
        eval_tasks = {
            "TWEET_EVAL_SENT": {
                "type": "single",
                "metric": "macro_f1",
                "train": "data/tweet_train.csv",
                "test": "data/tweet_test.csv",
            }
        }
    else:
        eval_tasks = download_and_prepare_eval_datasets()

    if exp_name in ("e0_faithful_v1", "e0_controlled_v1", "e1_objective_comparison", "v1_reproduction"):
        results = run_objective_comparison(config, eval_tasks_config=eval_tasks, output_dir=out_dir, device=device)
    elif exp_name == "e2_assignment_sweep":
        results = run_assignment_sweep(config, eval_tasks_config=eval_tasks, output_dir=out_dir, device=device)
    elif exp_name == "e3_nested_dimensions":
        results = run_nested_experiment(config, eval_tasks_config=eval_tasks, output_dir=out_dir, device=device)
    elif exp_name == "e4_gradient_analysis":
        checkpoints = config.get("analysis", {}).get("checkpoints", {})
        if not checkpoints:
            init_weights = out_dir / "initial_tinybert_weights.pt"
            checkpoints = {"initialization": str(init_weights)}
        results = run_gradient_analysis(config, checkpoints=checkpoints, output_dir=out_dir, device=device)
    else:
        raise ValueError(f"Unrecognized experiment name: {exp_name}")

    print(f"\nExperiment {exp_name} completed successfully! Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
