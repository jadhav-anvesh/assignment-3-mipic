"""Analyze and summarize Experiment E1 (Objective Comparison) results.

Computes multi-seed means, standard deviations, assigned-block performance,
and the four planned statistical contrasts:
1. Query Budget: C1 (global) vs C4 (global_matched_query)
2. Query Composition: C4 (global_matched_query) vs C2 (query_masked)
3. Negative Count: C2 (query_masked) vs C6 (query_masked_mixed_k3)
4. Negative Task Composition: C3 (task_isolated) vs C5 (query_masked_cross_k3)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
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

ASSIGNMENT: Dict[str, int] = {
    "TWEET_EVAL_SENT": 0,
    "BANKING77": 1,
    "MRPC": 2,
    "DEEHAN1866/WIC": 3,
    "DGEM_FORMAT": 4,
    "SICK": 5,
    "STSB": 6,
}

BLOCK_WIDTHS: List[str] = [
    "0:16 (16D)",
    "16:32 (16D)",
    "32:64 (32D)",
    "64:128 (64D)",
    "128:256 (128D)",
    "256:512 (256D)",
    "512:768 (256D)",
]

MODES: List[Tuple[str, str]] = [
    ("C1", "global"),
    ("C2", "query_masked"),
    ("C3", "task_isolated"),
    ("C4", "global_matched_query"),
    ("C5", "query_masked_cross_k3"),
    ("C6", "query_masked_mixed_k3"),
]

CONTRASTS: List[Tuple[str, str, str, str]] = [
    ("Contrast 1 (Query Budget)", "C1", "global", "C4", "global_matched_query", "28 queries vs 4 queries (all 27 negatives)"),
    ("Contrast 2 (Query Composition)", "C4", "global_matched_query", "C2", "query_masked", "4 mixed queries vs 4 task-assigned queries"),
    ("Contrast 3 (Negative Count)", "C2", "query_masked", "C6", "query_masked_mixed_k3", "27 negatives vs 3 mixed negatives (4 task queries)"),
    ("Contrast 4 (Negative Task Composition)", "C3", "task_isolated", "C5", "query_masked_cross_k3", "3 same-task negatives vs 3 cross-task negatives"),
]


def load_e1_data(base_dir: Path, seeds: List[int]) -> Dict[str, Dict[int, Dict[str, Dict[str, float]]]]:
    data: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for _, mode in MODES:
        data[mode] = {}
        for s in seeds:
            p = base_dir / mode / f"seed_{s}" / "metrics.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing expected metrics file: {p}")
            with open(p, "r", encoding="utf-8") as f:
                content = json.load(f)
                data[mode][s] = content["metrics_matrix"]
    return data


def analyze_e1(output_dir: str = "routing_v2/outputs/e1_objective_comparison", seeds: List[int] = [42, 123, 456]):
    base_dir = Path(output_dir)
    data = load_e1_data(base_dir, seeds)

    print("=" * 115)
    print("EXPERIMENT E1: OBJECTIVE DECOMPOSITION RESULTS (3 SEEDS: 42, 123, 456)")
    print("=" * 115)

    # 1. Assigned-Block Absolute Scores Table
    print("\n### 1. Assigned Block Performance Across All 6 Conditions (Mean ± Std Across Seeds)\n")
    header_cols = ["Task", "Assigned Blk"] + [f"{c_code} ({m_name[:9]})" for c_code, m_name in MODES]
    header_str = " | ".join(f"{col:<15}" for col in header_cols)
    print(header_str)
    print("-" * len(header_str))

    condition_means: Dict[str, List[float]] = {m: [] for _, m in MODES}

    for task in TASKS:
        blk = ASSIGNMENT[task]
        row = [f"{task:<15}", f"Blk {blk} ({BLOCK_WIDTHS[blk][:7]})"]
        for _, mode in MODES:
            scores = [data[mode][s][task][str(blk)] for s in seeds]
            m_val = float(np.mean(scores))
            s_val = float(np.std(scores))
            condition_means[mode].append(m_val)
            row.append(f"{m_val:.4f}±{s_val:.3f}")
        print(" | ".join(f"{val:<15}" for val in row))

    print("-" * len(header_str))
    macro_row = ["Macro Average", "All 7 Blocks"] + [
        f"{float(np.mean(condition_means[m])):.4f}" for _, m in MODES
    ]
    print(" | ".join(f"{val:<15}" for val in macro_row))
    print("=" * 115)

    # 2. Planned Contrasts Table
    print("\n### 2. Planned Contrasts (Paired Differences Across Seeds: A - B)\n")
    print("Hypotheses tested:")
    for name, cA, mA, cB, mB, desc in CONTRASTS:
        print(f"• {name}: {cA} ({mA}) minus {cB} ({mB}) -> {desc}")
    print()

    contrast_header = ["Task", "Assigned Blk"] + [f"{c[0].split('(')[1][:-1]}" for c in CONTRASTS]
    c_hdr_str = " | ".join(f"{col:<20}" for col in contrast_header)
    print(c_hdr_str)
    print("-" * len(c_hdr_str))

    contrast_macro_diffs: Dict[str, List[float]] = {c[0]: [] for c in CONTRASTS}

    for task in TASKS:
        blk = ASSIGNMENT[task]
        row = [f"{task:<20}", f"Blk {blk} ({BLOCK_WIDTHS[blk][:7]})"]
        for c_name, _, mA, _, mB, _ in CONTRASTS:
            # Paired differences per seed
            diffs = [data[mA][s][task][str(blk)] - data[mB][s][task][str(blk)] for s in seeds]
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs))
            contrast_macro_diffs[c_name].append(mean_d)
            row.append(f"{mean_d:+.4f}±{std_d:.3f}")
        print(" | ".join(f"{val:<20}" for val in row))

    print("-" * len(c_hdr_str))
    c_macro_row = ["Macro Paired Δ", "All 7 Blocks"] + [
        f"{float(np.mean(contrast_macro_diffs[c[0]])):+.4f} (std {float(np.std(contrast_macro_diffs[c[0]])):.4f})"
        for c in CONTRASTS
    ]
    print(" | ".join(f"{val:<20}" for val in c_macro_row))
    print("=" * 115)

    # 3. Individual Seed Pairing Breakdown for each contrast
    print("\n### 3. Individual Seed Breakdown for Planned Contrasts\n")
    for c_name, cA, mA, cB, mB, desc in CONTRASTS:
        print(f"--- {c_name}: {cA} ({mA}) vs {cB} ({mB}) [{desc}] ---")
        print(f"{'Task':<20} | {'Seed 42':<12} | {'Seed 123':<12} | {'Seed 456':<12} | {'Mean ± Std':<15}")
        print("-" * 75)
        for task in TASKS:
            blk = ASSIGNMENT[task]
            diffs = [data[mA][s][task][str(blk)] - data[mB][s][task][str(blk)] for s in seeds]
            m_d = float(np.mean(diffs))
            s_d = float(np.std(diffs))
            print(f"{task:<20} | {diffs[0]:>+10.4f} | {diffs[1]:>+10.4f} | {diffs[2]:>+10.4f} | {m_d:>+7.4f} ± {s_d:.4f}")
        macro_seed_diffs = [
            float(np.mean([data[mA][s][t][str(ASSIGNMENT[t])] - data[mB][s][t][str(ASSIGNMENT[t])] for t in TASKS]))
            for s in seeds
        ]
        print("-" * 75)
        print(f"{'Macro Average':<20} | {macro_seed_diffs[0]:>+10.4f} | {macro_seed_diffs[1]:>+10.4f} | {macro_seed_diffs[2]:>+10.4f} | {float(np.mean(macro_seed_diffs)):>+7.4f} ± {float(np.std(macro_seed_diffs)):.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze E1 objective comparison.")
    parser.add_argument("--output_dir", default="routing_v2/outputs/e1_objective_comparison", help="Output directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Evaluated seeds")
    args = parser.parse_args()

    analyze_e1(output_dir=args.output_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()

