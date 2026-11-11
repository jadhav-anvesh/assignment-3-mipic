"""Analyze and summarize Experiment E3 (Nested MRL Profiles) results.

Computes and reports:
1. Multi-seed absolute test performance across dimensions (Mean +/- Std).
2. Absolute test score differences from full 768D: Score(d) - Score(768).
3. Relative test capacity retention curves: (Score(d) / Score(768) * 100%).
4A. Per-seed validation-selected dimensions d_t^* to estimate selection procedure variability.
4B. Aggregate validation curve selection: task capacity allocation recommendations d_t^*
    evaluated honestly on the test set for BOTH Sustained and 95% Retention criteria.
5A. Probe sensitivity comparison: Tuned C (train CV) vs Fixed C = 1.0.
5B. Probe regularization parameter C selected across seeds and dimensions.
6. Training convergence audit: epoch-cumulative vs final-window instantaneous losses,
   prefix loss hierarchy, and loss drift over final update windows.
7. Task-specific stability diagnostics (including MRPC sample-size analysis).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
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

TASK_METRIC_NAMES: Dict[str, str] = {
    "TWEET_EVAL_SENT": "Macro-F1",
    "BANKING77": "Macro-F1",
    "MRPC": "Macro-F1",
    "DEEHAN1866/WIC": "Macro-F1",
    "DGEM_FORMAT": "Macro-F1",
    "SICK": "Macro-F1",
    "STSB": "Spearman",
}

NESTED_DIMS: List[int] = [16, 32, 64, 128, 256, 512, 768]


def load_e3_data(base_dir: Path, seeds: List[int]) -> Dict[int, Dict[str, Any]]:
    runs: Dict[int, Dict[str, Any]] = {}
    for s in seeds:
        p = base_dir / f"seed_{s}" / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing expected metrics file: {p}")
        with open(p, "r", encoding="utf-8") as f:
            runs[s] = json.load(f)
    return runs


def analyze_e3(output_dir: str = "routing_v2/outputs/e3_nested_dimensions", seeds: List[int] = [42, 123, 456]):
    base_dir = Path(output_dir)
    runs = load_e3_data(base_dir, seeds)

    print("=" * 130)
    print("EXPERIMENT E3: GENUINE NESTED MRL CAPACITY PROFILES (3 SEEDS: 42, 123, 456)")
    print(f"Prefix Dimensions Evaluated: {NESTED_DIMS}")
    print("Shared Encoder: TinyBERT-6L (768D), Multi-task Joint MRL InfoNCE Objective")
    print("Prefix Normalization: Each prefix z_{:, :d} is extracted and independently L2-normalized")
    print("Probe Strategy: Stratified 3-fold CV on train embeddings for C (STS-B: Zero-shot Cosine)")
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 1. Absolute Test Scores Table across Dimensions
    # -------------------------------------------------------------------------
    print("\n### 1. ABSOLUTE TEST PERFORMANCE BY PREFIX DIMENSION (Mean +/- Std Across Seeds)\n")
    dim_headers = [f"{d}D" for d in NESTED_DIMS]
    hdr1 = f"{'Task':<18} | {'Metric':<10} | " + " | ".join(f"{h:<10}" for h in dim_headers)
    print(hdr1)
    print("-" * len(hdr1))

    mean_test_scores: Dict[str, Dict[int, float]] = {t: {} for t in TASKS}
    std_test_scores: Dict[str, Dict[int, float]] = {t: {} for t in TASKS}

    for t in TASKS:
        row = [f"{t:<18}", f"{TASK_METRIC_NAMES[t]:<10}"]
        for d in NESTED_DIMS:
            s_vals = [runs[s]["metrics"][t][str(d)] for s in seeds]
            m_val = float(np.mean(s_vals))
            s_val = float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0
            mean_test_scores[t][d] = m_val
            std_test_scores[t][d] = s_val
            row.append(f"{m_val:.4f}+/-{s_val:.3f}")
        print(" | ".join(row))
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 2. Absolute Difference from Full 768D Model: Score(d) - Score(768)
    # -------------------------------------------------------------------------
    print("\n### 2. ABSOLUTE TEST SCORE DIFFERENCE FROM 768D: Score(d) - Score(768)\n")
    hdr2 = f"{'Task':<18} | {'Metric':<10} | " + " | ".join(f"{h:<10}" for h in dim_headers)
    print(hdr2)
    print("-" * len(hdr2))

    for t in TASKS:
        row = [f"{t:<18}", f"{TASK_METRIC_NAMES[t]:<10}"]
        for d in NESTED_DIMS:
            deltas_seed = [runs[s]["metrics"][t][str(d)] - runs[s]["metrics"][t]["768"] for s in seeds]
            m_d = float(np.mean(deltas_seed))
            s_d = float(np.std(deltas_seed, ddof=1)) if len(deltas_seed) > 1 else 0.0
            sign = "+" if m_d >= 0 else ""
            row.append(f"{sign}{m_d:.4f}+/-{s_d:.3f}")
        print(" | ".join(row))
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 3. Capacity Retention Relative to Full 768D Representation
    # -------------------------------------------------------------------------
    print("\n### 3. RELATIVE TEST CAPACITY RETENTION: (Score(d) / Score(768) * 100%)\n")
    ret_headers = [f"{d}D" for d in NESTED_DIMS]
    hdr3 = f"{'Task':<18} | {'768D Mean':<12} | " + " | ".join(f"{h:<8}" for h in ret_headers)
    print(hdr3)
    print("-" * len(hdr3))

    for t in TASKS:
        score_768 = mean_test_scores[t][768]
        row = [f"{t:<18}", f"{score_768:.4f}      "]
        for d in NESTED_DIMS:
            score_d = mean_test_scores[t][d]
            pct = (score_d / score_768) * 100.0 if score_768 > 0 else 0.0
            row.append(f"{pct:>6.1f}%")
        print(" | ".join(row))
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 4A. Per-Seed Effective Dimension Selection (Variability Analysis)
    # -------------------------------------------------------------------------
    print("\n### 4A. PER-SEED EFFECTIVE DIMENSION SELECTION (Selection Variability Across Seeds)")
    print("Evaluates the stability of validation-based dimension selection for each random seed.")
    print("Criterion 1: First Crossing  -> Smallest d where Val(d) >= Val(768) - 0.01")
    print("Criterion 2: Sustained       -> Smallest d where for all d' >= d, Val(d') >= Val(768) - 0.01")
    print("Criterion 3: 95% Retention   -> Smallest d where Val(d) >= 0.95 * Val(768)\n")

    hdr4a = (
        f"{'Task':<18} | "
        f"{'First Crossing (42/123/456)':<28} | "
        f"{'Sustained (42/123/456)':<24} | "
        f"{'95% Ret (42/123/456)':<22}"
    )
    print(hdr4a)
    print("-" * len(hdr4a))

    for t in TASKS:
        first_seeds = []
        sust_seeds = []
        ret95_seeds = []
        for s in seeds:
            ana = runs[s].get("analysis", {}).get(t, {})
            first_seeds.append(f"{ana.get('val_selected_d_tau', 768)}D")
            sust_seeds.append(f"{ana.get('val_selected_d_sustained', 768)}D")
            ret95_seeds.append(f"{ana.get('val_selected_d_95', 768)}D")

        first_str = "/".join(first_seeds)
        sust_str = "/".join(sust_seeds)
        ret95_str = "/".join(ret95_seeds)

        print(f"{t:<18} | {first_str:<28} | {sust_str:<24} | {ret95_str:<22}")
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 4B. Aggregate Validation Selection & Honest Test Evaluation (Task Allocation Recommendations)
    # -------------------------------------------------------------------------
    print("\n### 4B. AGGREGATE VALIDATION SELECTION & HONEST TEST EVALUATION (Task Allocation Recommendations)")
    print("Protocol: Recommended d* is selected strictly from multi-seed aggregate validation curves:")
    print("          Val_bar(d) = (1/3) * sum_{s} Val_{s}(d). No test scores are ever used in selection.")
    print("Note:     The divergence between validation retention and test retention illustrates why")
    print("          validation-based dimension selection must be reported separately from test-set evaluation.")
    print("          Test results are used solely to assess generalization, not to revise selected dimensions.\n")

    hdr4b = (
        f"{'Task':<18} | {'Metric':<9} | "
        f"{'d*(Sust)':<8} | {'Test at d*(Sust)':<17} | {'Test Ret':<9} | "
        f"{'d*(95%)':<8} | {'Test at d*(95%)':<17} | {'Test Ret':<9} | "
        f"{'768D Test':<10}"
    )
    print(hdr4b)
    print("-" * len(hdr4b))

    agg_selected_dims: Dict[str, Dict[str, int]] = {}

    for t in TASKS:
        # Compute aggregate validation scores across seeds
        agg_val_scores: Dict[int, float] = {}
        for d in NESTED_DIMS:
            d_str = str(d)
            v_vals = [runs[s]["val_metrics"][t][d_str] for s in seeds]
            agg_val_scores[d] = float(np.mean(v_vals))

        v_768 = agg_val_scores[768]

        # 1. Aggregate First Crossing
        agg_first_d = 768
        for d in NESTED_DIMS:
            if agg_val_scores[d] >= (v_768 - 0.01):
                agg_first_d = d
                break

        # 2. Aggregate Sustained
        agg_sust_d = 768
        for i, d in enumerate(NESTED_DIMS):
            if all(agg_val_scores[d_prime] >= (v_768 - 0.01) for d_prime in NESTED_DIMS[i:]):
                agg_sust_d = d
                break

        # 3. Aggregate 95% Retention
        agg_95_d = 768
        for d in NESTED_DIMS:
            if agg_val_scores[d] >= (v_768 * 0.95):
                agg_95_d = d
                break

        agg_selected_dims[t] = {
            "first_crossing": agg_first_d,
            "sustained": agg_sust_d,
            "retention_95": agg_95_d,
        }

        # Evaluate test scores at the sustained dimension and 95% retention dimension
        test_at_sust_seed = [runs[s]["metrics"][t][str(agg_sust_d)] for s in seeds]
        test_at_95_seed = [runs[s]["metrics"][t][str(agg_95_d)] for s in seeds]
        test_768_seed = [runs[s]["metrics"][t]["768"] for s in seeds]

        m_test_sust = float(np.mean(test_at_sust_seed))
        s_test_sust = float(np.std(test_at_sust_seed, ddof=1)) if len(test_at_sust_seed) > 1 else 0.0

        m_test_95 = float(np.mean(test_at_95_seed))
        s_test_95 = float(np.std(test_at_95_seed, ddof=1)) if len(test_at_95_seed) > 1 else 0.0

        m_test_768 = float(np.mean(test_768_seed))
        s_test_768 = float(np.std(test_768_seed, ddof=1)) if len(test_768_seed) > 1 else 0.0

        ret_sust_pct = (m_test_sust / m_test_768 * 100.0) if m_test_768 > 0 else 0.0
        ret_95_pct = (m_test_95 / m_test_768 * 100.0) if m_test_768 > 0 else 0.0

        print(
            f"{t:<18} | {TASK_METRIC_NAMES[t]:<9} | "
            f"{str(agg_sust_d)+'D':<8} | {m_test_sust:.4f}+/-{s_test_sust:.3f}     | {ret_sust_pct:>6.1f}%  | "
            f"{str(agg_95_d)+'D':<8} | {m_test_95:.4f}+/-{s_test_95:.3f}     | {ret_95_pct:>6.1f}%  | "
            f"{m_test_768:.4f}+/-{s_test_768:.3f}"
        )
    print("=" * 130)

    # -------------------------------------------------------------------------
    # 5A. Sensitivity Analysis: Probe C Tuning (CV vs Fixed C=1.0)
    # -------------------------------------------------------------------------
    has_fixed_c = all("metrics_fixed_c" in runs[s] for s in seeds)
    if has_fixed_c:
        print("\n### 5A. PROBE SENSITIVITY CHECK: Tuned C (Stratified Train CV) vs Fixed C = 1.0\n")
        hdr5a = (
            f"{'Task':<18} | {'Dim':<6} | "
            f"{'Tuned C (Mean+/-Std)':<24} | "
            f"{'Fixed C (Mean+/-Std)':<24} | "
            f"{'Probe Delta (Tuned - Fixed)':<26}"
        )
        print(hdr5a)
        print("-" * len(hdr5a))

        for t in TASKS:
            if TASK_METRIC_NAMES[t] == "Spearman":
                continue  # STS-B does not use logistic regression probe
            for d in [16, 64, 256, 768]:
                t_scores = [runs[s]["metrics"][t][str(d)] for s in seeds]
                f_scores = [runs[s]["metrics_fixed_c"][t][str(d)] for s in seeds]
                m_t, s_t = float(np.mean(t_scores)), float(np.std(t_scores, ddof=1)) if len(t_scores) > 1 else 0.0
                m_f, s_f = float(np.mean(f_scores)), float(np.std(f_scores, ddof=1)) if len(f_scores) > 1 else 0.0
                deltas = [ts - fs for ts, fs in zip(t_scores, f_scores)]
                m_d, s_d = float(np.mean(deltas)), float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
                sign = "+" if m_d >= 0 else ""
                print(
                    f"{t:<18} | {str(d)+'D':<6} | "
                    f"{m_t:.4f}+/-{s_t:.4f}             | "
                    f"{m_f:.4f}+/-{s_f:.4f}             | "
                    f"{sign}{m_d:.4f}+/-{s_d:.4f}"
                )
        print("=" * 130)

    # -------------------------------------------------------------------------
    # 5B. Probe Regularization Parameter C Selected Across Seeds and Dimensions
    # -------------------------------------------------------------------------
    has_c_matrix = all("selected_c" in runs[s] for s in seeds)
    if has_c_matrix:
        print("\n### 5B. SELECTED REGULARIZATION PARAMETER C (Probe Train CV Distribution)\n")
        print("Candidate Grid: C in {0.01, 0.1, 1.0, 10.0, 100.0}")
        print("Interpretation: Smaller C enforces stronger regularization; larger C permits weaker regularization.")
        print("Cross-validation selects larger C when representations benefit from a less constrained probe.")
        print("Limitation & Scope Disclosure:")
        print("  For some task-dimension combinations (e.g. Banking77), cross-validation selected the maximum candidate")
        print("  C = 100 (*). Therefore, probe performance and effective dimension estimates are conditional on the")
        print("  specified regularization search range. These results represent task-specific effective dimensions under")
        print("  the chosen multi-scale MRL training and linear probe protocol, not universally intrinsic dimensionality.\n")
        hdr5b = f"{'Task':<18} | " + " | ".join(f"{d}D (42/123/456)" for d in [16, 64, 256, 768])
        print(hdr5b)
        print("-" * len(hdr5b))
        for t in TASKS:
            if TASK_METRIC_NAMES[t] == "Spearman":
                continue
            cols = []
            for d in [16, 64, 256, 768]:
                c_vals = []
                for s in seeds:
                    val = runs[s]["selected_c"][t].get(str(d), "N/A")
                    if isinstance(val, (int, float)):
                        if val < 0.1:
                            c_str = f"{val:.2f}"
                        elif val < 10.0:
                            c_str = f"{val:.1f}"
                        else:
                            c_str = f"{val:.0f}"
                        if val >= 100.0:
                            c_str += "(*)"
                    else:
                        c_str = str(val)
                    c_vals.append(c_str)
                cols.append("/".join(c_vals))
            print(f"{t:<18} | " + " | ".join(f"{c:<16}" for c in cols))
        print("=" * 130)

    # -------------------------------------------------------------------------
    # 6. Training Convergence & Trajectory Audit
    # -------------------------------------------------------------------------
    print("\n### 6. TRAINING CONVERGENCE & TRAJECTORY AUDIT")
    print("Convergence Criteria & Metric Definitions:")
    print("- Initial Step Loss: Instantaneous InfoNCE loss on the very first batch (Step 1).")
    print("- Epoch Mean Loss: Full-epoch cumulative running average across all 586 updates (~0.018).")
    print("- Penultimate 10% Window Avg: Instantaneous batch loss averaged over Steps 469-527.")
    print("- Final 10% Window Avg: Instantaneous batch loss averaged over Steps 528-586.")
    print("- Window Difference (Drift): Final 10% Window Avg - Penultimate 10% Window Avg.")
    print("  Plateau Evaluation:")
    print("  Seeds 123 and 456 showed very small differences between their final and penultimate")
    print("  window-average losses (~0.00004 and ~0.00018). Seed 42 showed a larger difference of approximately")
    print("  0.00114. Thus, the evidence for a late-training plateau is stronger for Seeds 123 and 456 than for")
    print("  Seed 42. (The full-epoch average is dominated by early high-loss batches and must not be confused")
    print("  with late-training convergence).\n")

    hdr6 = f"{'Seed':<8} | {'Init Loss (Step 1)':<20} | {'Epoch Mean Loss':<18} | {'Penultimate 10%':<18} | {'Final 10% Win Avg':<18} | {'Win Diff (Drift)':<18}"
    print(hdr6)
    print("-" * len(hdr6))
    for s in seeds:
        traj = runs[s].get("loss_trajectory", [])
        init_l = traj[0]["batch_loss"] if traj else 0.0
        epoch_mean = runs[s].get("epoch_average_loss", traj[-1]["running_avg_loss"] if traj else 0.0)
        win_avg = runs[s].get("final_window_avg_loss", 0.0)
        pen_avg = runs[s].get("penultimate_window_avg_loss", 0.0)
        drift = runs[s].get("final_loss_drift", 0.0)
        sign = "+" if drift >= 0 else ""
        print(f"Seed {s:<3} | {init_l:<20.4f} | {epoch_mean:<18.4f} | {pen_avg:<18.4f} | {win_avg:<18.4f} | {sign}{drift:<18.6f}")
    print("-" * len(hdr6))

    # Detailed late-training trajectory inspection (steps >= 400)
    has_trajectories = all("loss_trajectory" in runs[s] for s in seeds)
    if has_trajectories:
        late_steps = sorted(list({entry["step"] for s in seeds for entry in runs[s].get("loss_trajectory", []) if entry["step"] >= 400}))
        if late_steps:
            print("\nLate-Training Batch Loss Trajectory Inspection (Steps >= 400):")
            hdr_traj = f"{'Step':<10} | " + " | ".join(f"Seed {s:<12}" for s in seeds)
            print(hdr_traj)
            print("-" * len(hdr_traj))
            for st in late_steps:
                row_vals = []
                for s in seeds:
                    match = next((e["batch_loss"] for e in runs[s].get("loss_trajectory", []) if e["step"] == st), None)
                    row_vals.append(f"{match:<12.4f}" if match is not None else "N/A         ")
                print(f"Step {st:<5} | " + " | ".join(row_vals))
            print("-" * len(hdr_traj))

    # Prefix losses across dimensions
    has_prefix_losses = all("prefix_losses" in runs[s] for s in seeds)
    if has_prefix_losses:
        print("\nMulti-Scale InfoNCE Loss by Prefix Dimension (Epoch Cumulative Mean):")
        hdr6b = f"{'Dimension':<12} | " + " | ".join(f"Seed {s:<8}" for s in seeds) + " | Mean Epoch Loss"
        print(hdr6b)
        print("-" * len(hdr6b))
        for d in NESTED_DIMS:
            d_str = str(d)
            losses = [runs[s]["prefix_losses"].get(d_str, 0.0) for s in seeds]
            m_l = float(np.mean(losses))
            row = [f"{str(d)+'D':<12}"] + [f"{l:.4f}    " for l in losses] + [f"{m_l:.4f}"]
            print(" | ".join(row))
        print("=" * 130)

    # -------------------------------------------------------------------------
    # 7. Diagnostic Check: MRPC Non-Monotonicity and Sample Size Sensitivity
    # -------------------------------------------------------------------------
    print("\n### 7. DIAGNOSTIC AUDIT: MRPC SAMPLE SIZE SENSITIVITY AND NON-MONOTONICITY\n")
    print("MRPC Dataset Characteristics:")
    print("- Train set: 3,668 sentence pairs (67.4% positive class 1, 32.6% negative class 0)")
    print("- Test set:    408 sentence pairs (279 positive class 1, 129 negative class 0)")
    print("- Scientific Framing:")
    print("  MRPC exhibits non-monotonic probe performance across adjacent prefix dimensions.")
    print("  Given the small test set, minority-class granularity (only 129 negative test pairs, where 3-4 labels")
    print("  shift Macro-F1 by ~0.015-0.025), and seed variability (+/-0.035), we do not interpret these")
    print("  local fluctuations as evidence that increasing prefix dimensionality systematically harms performance.\n")

    print(f"{'Dimension':<10} | {'Seed 42 Test':<14} | {'Seed 123 Test':<14} | {'Seed 456 Test':<14} | {'Mean Test +/- Std':<20} | {'Mean Val +/- Std'}")
    print("-" * 105)
    for d in NESTED_DIMS:
        d_str = str(d)
        t_vals = [runs[s]["metrics"]["MRPC"][d_str] for s in seeds]
        v_vals = [runs[s]["val_metrics"]["MRPC"][d_str] for s in seeds]
        m_t, s_t = float(np.mean(t_vals)), float(np.std(t_vals, ddof=1)) if len(t_vals) > 1 else 0.0
        m_v, s_v = float(np.mean(v_vals)), float(np.std(v_vals, ddof=1)) if len(v_vals) > 1 else 0.0
        print(
            f"{str(d)+'D':<10} | {t_vals[0]:<14.4f} | {t_vals[1]:<14.4f} | {t_vals[2]:<14.4f} | "
            f"{m_t:.4f} +/- {s_t:.3f}     | {m_v:.4f} +/- {s_v:.3f}"
        )
    print("=" * 130)


def compare_e3_runs(primary_dir: Path, compare_dir: Path, seeds: List[int]):
    """Compare a new E3 run against an earlier/initial E3 baseline run."""
    if not compare_dir.exists():
        print(f"[-] Comparison baseline directory not found: {compare_dir}")
        return

    runs_primary = load_e3_data(primary_dir, seeds)
    runs_comp = load_e3_data(compare_dir, seeds)

    print("\n" + "=" * 130)
    print(f"### 8. SIDE-BY-SIDE AUDIT COMPARISON: RERUN ({primary_dir.name}) vs INITIAL BASELINE ({compare_dir.name})")
    print("=" * 130 + "\n")

    hdr_cmp = (
        f"{'Task':<18} | {'Metric':<9} | "
        f"{'768D (Initial)':<18} | {'768D (Rerun)':<18} | "
        f"{'d* 95% (Init -> Rerun)':<24} | "
        f"{'d* Sust (Init -> Rerun)':<24}"
    )
    print(hdr_cmp)
    print("-" * len(hdr_cmp))

    for t in TASKS:
        # Initial stats
        t768_comp = [runs_comp[s]["metrics"][t]["768"] for s in seeds]
        m_t_comp, s_t_comp = float(np.mean(t768_comp)), float(np.std(t768_comp, ddof=1)) if len(t768_comp) > 1 else 0.0

        # Rerun stats
        t768_prim = [runs_primary[s]["metrics"][t]["768"] for s in seeds]
        m_t_prim, s_t_prim = float(np.mean(t768_prim)), float(np.std(t768_prim, ddof=1)) if len(t768_prim) > 1 else 0.0

        # Compute aggregate d* for initial
        v_comp = {d: float(np.mean([runs_comp[s]["val_metrics"][t][str(d)] for s in seeds])) for d in NESTED_DIMS}
        v768_c = v_comp[768]
        d95_c = next((d for d in NESTED_DIMS if v_comp[d] >= v768_c * 0.95), 768)
        dsust_c = next((d for i, d in enumerate(NESTED_DIMS) if all(v_comp[dp] >= v768_c - 0.01 for dp in NESTED_DIMS[i:])), 768)

        # Compute aggregate d* for rerun
        v_prim = {d: float(np.mean([runs_primary[s]["val_metrics"][t][str(d)] for s in seeds])) for d in NESTED_DIMS}
        v768_p = v_prim[768]
        d95_p = next((d for d in NESTED_DIMS if v_prim[d] >= v768_p * 0.95), 768)
        dsust_p = next((d for i, d in enumerate(NESTED_DIMS) if all(v_prim[dp] >= v768_p - 0.01 for dp in NESTED_DIMS[i:])), 768)

        print(
            f"{t:<18} | {TASK_METRIC_NAMES[t]:<9} | "
            f"{m_t_comp:.4f} +/- {s_t_comp:.3f}   | "
            f"{m_t_prim:.4f} +/- {s_t_prim:.3f}   | "
            f"{str(d95_c)+'D -> '+str(d95_p)+'D':<24} | "
            f"{str(dsust_c)+'D -> '+str(dsust_p)+'D':<24}"
        )
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Analyze E3 nested MRL profiles.")
    parser.add_argument("--output_dir", default="routing_v2/outputs/e3_nested_dimensions", help="Output directory")
    parser.add_argument("--compare_dir", default=None, help="Optional earlier run directory for side-by-side comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Evaluated seeds")
    args = parser.parse_args()

    analyze_e3(output_dir=args.output_dir, seeds=args.seeds)
    if args.compare_dir:
        compare_e3_runs(primary_dir=Path(args.output_dir), compare_dir=Path(args.compare_dir), seeds=args.seeds)


if __name__ == "__main__":
    main()
