"""Generate publication-ready figures for Experiment E3 (Nested MRL Profiles).

Outputs:
1. e3_validation_retention_curves.png: Validation capacity retention curves per task with seed lines.
2. e3_test_retention_curves.png: Test capacity retention curves per task with seed lines and d* markers.
3. e3_probe_c_sensitivity.png: Comparison of Tuned C vs Fixed C across prefix dimensions.
4. e3_training_loss_trajectory.png: InfoNCE multi-scale loss convergence trajectories across steps.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np

TASKS: List[str] = [
    "TWEET_EVAL_SENT",
    "BANKING77",
    "MRPC",
    "DEEHAN1866/WIC",
    "DGEM_FORMAT",
    "SICK",
    "STSB",
]

TASK_TITLES: Dict[str, str] = {
    "TWEET_EVAL_SENT": "TweetEval (Sentiment)",
    "BANKING77": "Banking77 (Fine-Grained Intent)",
    "MRPC": "MRPC (Paraphrase)",
    "DEEHAN1866/WIC": "WiC (Word-in-Context)",
    "DGEM_FORMAT": "DGEM (Directed Entailment)",
    "SICK": "SICK (Entailment)",
    "STSB": "STS-B (Semantic Similarity)",
}

NESTED_DIMS: List[int] = [16, 32, 64, 128, 256, 512, 768]
SEED_COLORS: Dict[int, str] = {42: "#1f77b4", 123: "#2ca02c", 456: "#ff7f0e"}


def load_e3_data(base_dir: Path, seeds: List[int]) -> Dict[int, Dict[str, Any]]:
    runs: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        p = base_dir / f"seed_{s}" / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing expected metrics file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            runs[s] = json.load(f)
    return runs


def plot_validation_retention(runs: Dict[int, Dict[str, Any]], seeds: List[int], out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, t in enumerate(TASKS):
        ax = axes_flat[i]
        # Aggregate validation scores
        agg_val = []
        for d in NESTED_DIMS:
            vals = [runs[s]["val_metrics"][t][str(d)] for s in seeds]
            agg_val.append(float(np.mean(vals)))
        agg_val_768 = agg_val[-1]
        agg_ret = [v / agg_val_768 * 100.0 if agg_val_768 > 0 else 0.0 for v in agg_val]

        # Individual seed curves
        for s in seeds:
            s_val = [runs[s]["val_metrics"][t][str(d)] for d in NESTED_DIMS]
            s_768 = s_val[-1]
            s_ret = [v / s_768 * 100.0 if s_768 > 0 else 0.0 for v in s_val]
            ax.plot(
                NESTED_DIMS,
                s_ret,
                linestyle="--",
                alpha=0.45,
                color=SEED_COLORS.get(s, "gray"),
                label=f"Seed {s}",
            )

        # Bold aggregate curve
        ax.plot(NESTED_DIMS, agg_ret, color="black", linewidth=2.2, label="Mean Val Retention")

        # Threshold lines
        ax.axhline(100.0, color="gray", linestyle=":", alpha=0.6)
        ax.axhline(95.0, color="red", linestyle="--", alpha=0.5, label="95% Threshold")

        ax.set_title(TASK_TITLES[t], fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(NESTED_DIMS)
        ax.set_xticklabels([f"{d}D" for d in NESTED_DIMS], rotation=45)
        ax.set_ylabel("Validation Retention (%)")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8)

    # Blank out 8th subplot
    axes_flat[7].axis("off")

    fig.suptitle(
        "Experiment E3: Task-Specific Capacity Retention on Validation Set (Pre-Test Selection)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved validation retention plot to: {out_path}")


def plot_test_retention(runs: Dict[int, Dict[str, Any]], seeds: List[int], out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, t in enumerate(TASKS):
        ax = axes_flat[i]
        # Aggregate test scores
        agg_test = []
        for d in NESTED_DIMS:
            vals = [runs[s]["metrics"][t][str(d)] for s in seeds]
            agg_test.append(float(np.mean(vals)))
        agg_test_768 = agg_test[-1]
        agg_ret = [v / agg_test_768 * 100.0 if agg_test_768 > 0 else 0.0 for v in agg_test]

        # Determine validation-selected dimensions (strictly from val_metrics)
        agg_val = [float(np.mean([runs[s]["val_metrics"][t][str(d)] for s in seeds])) for d in NESTED_DIMS]
        val_768 = agg_val[-1]
        agg_sust_d = 768
        for idx, d in enumerate(NESTED_DIMS):
            if all(agg_val[k] >= (val_768 - 0.01) for k in range(idx, len(NESTED_DIMS))):
                agg_sust_d = d
                break
        agg_95_d = 768
        for d, v in zip(NESTED_DIMS, agg_val):
            if v >= (val_768 * 0.95):
                agg_95_d = d
                break

        # Individual seed curves
        for s in seeds:
            s_test = [runs[s]["metrics"][t][str(d)] for d in NESTED_DIMS]
            s_768 = s_test[-1]
            s_ret = [v / s_768 * 100.0 if s_768 > 0 else 0.0 for v in s_test]
            ax.plot(
                NESTED_DIMS,
                s_ret,
                linestyle="--",
                alpha=0.35,
                color=SEED_COLORS.get(s, "gray"),
                label=f"Seed {s}" if i == 0 else "",
            )

        # Bold aggregate curve
        ax.plot(NESTED_DIMS, agg_ret, color="darkblue", linewidth=2.2, label="Mean Test Retention" if i == 0 else "")

        # Mark validation-selected d*(Sustained) and d*(95% Val) on the test curve
        test_ret_sust = agg_ret[NESTED_DIMS.index(agg_sust_d)]
        test_ret_95 = agg_ret[NESTED_DIMS.index(agg_95_d)]

        ax.axvline(agg_sust_d, color="#9467bd", linestyle=":", alpha=0.8, linewidth=1.5)
        ax.plot(
            agg_sust_d,
            test_ret_sust,
            marker="*",
            markersize=13,
            color="#9467bd",
            linestyle="None",
            label=f"Val d*(Sust)={agg_sust_d}D ({test_ret_sust:.1f}%)" if i == 0 else "",
            zorder=5,
        )

        ax.axvline(agg_95_d, color="#2ca02c", linestyle="--", alpha=0.8, linewidth=1.5)
        ax.plot(
            agg_95_d,
            test_ret_95,
            marker="^",
            markersize=9,
            color="#2ca02c",
            linestyle="None",
            label=f"Val d*(95%)={agg_95_d}D ({test_ret_95:.1f}%)" if i == 0 else "",
            zorder=5,
        )

        # Reference lines
        ax.axhline(100.0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(95.0, color="crimson", linestyle="--", alpha=0.5, label="95% Target" if i == 0 else "")

        ax.set_title(TASK_TITLES[t], fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(NESTED_DIMS)
        ax.set_xticklabels([f"{d}D" for d in NESTED_DIMS], rotation=45)
        ax.set_ylabel("Test Retention (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)

    axes_flat[7].axis("off")

    fig.suptitle(
        "Experiment E3: Task-Specific Test Capacity Retention & Validation-Selected Dimension Generalization",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved test retention plot to: {out_path}")


def plot_probe_sensitivity(runs: Dict[int, Dict[str, Any]], seeds: List[int], out_path: Path):
    has_fixed_c = all("metrics_fixed_c" in runs[s] for s in seeds)
    if not has_fixed_c:
        print("[!] No fixed C metrics found; skipping probe sensitivity plot.")
        return

    cls_tasks = [t for t in TASKS if t != "STSB"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes_flat = axes.flatten()

    for i, t in enumerate(cls_tasks):
        ax = axes_flat[i]
        tuned_means = [float(np.mean([runs[s]["metrics"][t][str(d)] for s in seeds])) for d in NESTED_DIMS]
        fixed_means = [float(np.mean([runs[s]["metrics_fixed_c"][t][str(d)] for s in seeds])) for d in NESTED_DIMS]

        ax.plot(NESTED_DIMS, tuned_means, "o-", color="#1f77b4", linewidth=2.0, label="Tuned C (Train CV)")
        ax.plot(NESTED_DIMS, fixed_means, "s--", color="#d62728", linewidth=1.8, label="Fixed C = 1.0")

        # Fill difference
        ax.fill_between(NESTED_DIMS, tuned_means, fixed_means, color="#1f77b4", alpha=0.15)

        ax.set_title(TASK_TITLES[t], fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(NESTED_DIMS)
        ax.set_xticklabels([f"{d}D" for d in NESTED_DIMS], rotation=45)
        ax.set_ylabel("Macro-F1")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Probe Regularization Sensitivity: Tuned C (Stratified Train CV) vs Fixed C = 1.0",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved probe sensitivity plot to: {out_path}")


def plot_loss_convergence(runs: Dict[int, Dict[str, Any]], seeds: List[int], out_path: Path):
    has_traj = any("loss_trajectory" in runs[s] and len(runs[s]["loss_trajectory"]) > 0 for s in seeds)
    if not has_traj:
        print("[!] No loss trajectory data found; skipping convergence plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for s in seeds:
        traj = runs[s].get("loss_trajectory", [])
        if not traj:
            continue
        steps = [entry["step"] for entry in traj]
        losses = [entry["batch_loss"] for entry in traj]
        running_avgs = [entry["running_avg_loss"] for entry in traj]
        c = SEED_COLORS.get(s, "gray")

        ax1.plot(steps, losses, alpha=0.35, color=c, linestyle=":")
        ax1.plot(steps, running_avgs, color=c, linewidth=1.8, label=f"Seed {s} Running Avg")

    ax1.set_title("Multi-Scale InfoNCE Loss Trajectory (All 586 Steps)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("InfoNCE Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Bar plot of final prefix losses by dimension
    x_pos = np.arange(len(NESTED_DIMS))
    width = 0.25
    for idx, s in enumerate(seeds):
        losses = [runs[s].get("prefix_losses", {}).get(str(d), 0.0) for d in NESTED_DIMS]
        ax2.bar(x_pos + (idx - 1) * width, losses, width, color=SEED_COLORS.get(s, "gray"), label=f"Seed {s}")

    ax2.set_title("Epoch Mean InfoNCE Loss by Prefix Dimension", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{d}D" for d in NESTED_DIMS])
    ax2.set_xlabel("Prefix Dimension")
    ax2.set_ylabel("Mean InfoNCE Loss")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right")

    fig.suptitle("Experiment E3: Multi-Scale InfoNCE Training Convergence & Prefix Loss Hierarchy", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Saved loss convergence plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication plots for E3.")
    parser.add_argument("--output_dir", default="routing_v2/outputs/e3_nested_dimensions", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Evaluated seeds")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    runs = load_e3_data(base_dir, args.seeds)

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_validation_retention(runs, args.seeds, figures_dir / "e3_validation_retention_curves.png")
    plot_test_retention(runs, args.seeds, figures_dir / "e3_test_retention_curves.png")
    plot_probe_sensitivity(runs, args.seeds, figures_dir / "e3_probe_c_sensitivity.png")
    plot_loss_convergence(runs, args.seeds, figures_dir / "e3_training_loss_trajectory.png")


if __name__ == "__main__":
    main()

