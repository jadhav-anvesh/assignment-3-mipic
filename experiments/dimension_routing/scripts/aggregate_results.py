"""Aggregate multi-seed evaluation results and compute statistical confidence."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List
from routing_v2.src.evaluation.metrics import compute_summary_statistics


def aggregate_experiment_runs(output_dir: str | Path) -> Dict[str, Any]:
    """Traverse run directories, load metrics.json, and aggregate per-task scores."""
    out_path = Path(output_dir)
    results_by_mode: Dict[str, Dict[str, List[float]]] = {}

    for metrics_file in out_path.rglob("metrics.json"):
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mode = data.get("mode", "default")
        matrix = data.get("metrics_matrix") or data.get("metrics", {})

        for task, scores in matrix.items():
            results_by_mode.setdefault(mode, {}).setdefault(task, [])
            # Store the highest dimension or assigned block score
            first_score = list(scores.values())[0] if scores else 0.0
            results_by_mode[mode][task].append(float(first_score))

    summary = {}
    for mode, task_dict in results_by_mode.items():
        summary[mode] = {}
        for task, scores in task_dict.items():
            summary[mode][task] = compute_summary_statistics(scores)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument("--output_dir", required=True, help="Directory containing run outputs.")
    args = parser.parse_args()

    summary = aggregate_experiment_runs(args.output_dir)
    print("=" * 60)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
