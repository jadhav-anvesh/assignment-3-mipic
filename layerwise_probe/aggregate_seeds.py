"""Cross-seed summary of the layerwise_probe results for seeds 42, 1 and 2.

Reads the per-seed result files in layerwise_probe/results/ (and each seed's
runs/seed<N>/tweeteval_results.json for the selected epochs), writes one CSV
per table as results/cross_seed_*.csv, and writes layerwise_probe/CROSS_SEED.md.

For a quantity measured once per seed, the summary is the mean, the sample
standard deviation (ddof=1), the min and max, and how many seeds share the
sign of the mean. With three seeds no seed-level p-values or t-intervals are
reported. Per-seed significance always comes from the paired bootstraps over
test rows (or STS pairs) that the per-seed scripts already ran.

Decision rules (stated before the seed 1 and 2 results were looked at):
- H11: SUPPORTED if, in all three seeds, mipic beats both mrl and base, and
  mrl_pic beats mrl, at (1,16) and (2,32), with bootstrap CIs excluding zero.
  PARTIALLY SUPPORTED if this holds in exactly two seeds. NOT SUPPORTED otherwise.
- Finding 1: HOLDS if, in every seed, no trained model is significantly above
  base at the pooler row under both the default probe (check 4) and the tuned
  probe (check 6). WEAKENED if that happens in at least one cell in one seed.
  OVERTURNED if it happens in two or more seeds. For the tuned probe,
  "significantly above" means the full-size paired bootstrap CI is above zero,
  or the model beats base on every subset at a smaller probe size.
- H16 early synergy: HOLDS if the raw [CLS] interaction at (1,16) has a CI
  excluding zero and is positive in all three seeds.

Example, from the repository root:
    python layerwise_probe/aggregate_seeds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEEDS = {"s42": 42, "s1": 1, "s2": 2}
MODELS = ("mrl", "mrl_pic", "mipic", "mrl_sia")
LABEL = {"base": "base", "mrl": "MRL", "mrl_pic": "MRL+PIC", "mipic": "MIPIC", "mrl_sia": "MRL+SIA"}
DIMS = (16, 32, 64, 128, 256, 512, 768)
PIC = [(1, 16), (2, 32), (3, 64), (4, 256), (5, 512), (6, 768)]
H11_CELLS = [(1, 16), (2, 32)]
H11_TESTS = [("mipic", "mrl"), ("mipic", "base"), ("mrl_pic", "mrl")]


# ---------------------------------------------------------------- helpers

def summarise(values: List[float]) -> Dict[str, object]:
    v = np.asarray(values, dtype=float)
    mean = float(v.mean())
    sign = np.sign(mean)
    return {"mean": mean, "std": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
            "min": float(v.min()), "max": float(v.max()),
            "same_sign": int((np.sign(v) == sign).sum()) if sign != 0 else 0, "n_seeds": len(v)}


def cell(s: Dict[str, object], digits: int = 2) -> str:
    return (f"{s['mean']:.{digits}f} ± {s['std']:.{digits}f} "
            f"[{s['min']:.{digits}f}, {s['max']:.{digits}f}] {s['same_sign']}/{s['n_seeds']}")


def ci(diff: float, low: float, high: float) -> str:
    star = "*" if (low > 0 or high < 0) else ""
    return f"{diff:+.2f} [{low:+.2f}, {high:+.2f}]{star}"


def name(model: str, tag: str) -> str:
    return "base" if model == "base" else f"{model}_{tag}"


class Results:
    def __init__(self, results_dir: Path, runs_dir: Path):
        self.dir = results_dir
        self.runs = runs_dir

    def csv(self, filename: str) -> pd.DataFrame:
        frame = pd.read_csv(self.dir / filename)
        if "layer" in frame.columns:
            frame["layer"] = frame["layer"].astype(str)
        return frame

    def grid(self, model: str, tag: str) -> pd.DataFrame:
        g = self.csv(f"{name(model, tag)}_grid.csv")
        g["f1"] = g["f1"] * 100
        return g

    def value(self, frame: pd.DataFrame, layer, dim, col: str) -> float:
        return float(frame[(frame["layer"] == str(layer)) & (frame["dim"] == dim)][col].iloc[0])

    def boot(self, a: str, b: str, tag: str) -> pd.DataFrame:
        return self.csv(f"bootstrap_{name(a, tag)}_vs_{name(b, tag)}.csv")


def md_table(frame: pd.DataFrame) -> str:
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- tables

def table_pooler_f1(r: Results):
    rows, show = [], []
    for m in ("base",) + MODELS:
        disp = {"model": LABEL[m]}
        for d in DIMS:
            vals = [r.value(r.grid(m, t), "pooler", d, "f1") for t in SEEDS]
            s = summarise(vals)
            rows.append({"model": m, "dim": d, **{f"seed{SEEDS[t]}": v for t, v in zip(SEEDS, vals)}, **s})
            disp[str(d)] = f"{s['mean']:.2f} ± {s['std']:.2f}" if m != "base" else f"{s['mean']:.2f}"
        show.append(disp)
    return pd.DataFrame(rows), pd.DataFrame(show)


def table_h11(r: Results):
    f1_rows, f1_show = [], []
    for m in ("base",) + MODELS:
        disp = {"model": LABEL[m]}
        for layer, d in PIC:
            vals = [r.value(r.grid(m, t), layer, d, "f1") for t in SEEDS]
            s = summarise(vals)
            f1_rows.append({"model": m, "layer": layer, "dim": d,
                            **{f"seed{SEEDS[t]}": v for t, v in zip(SEEDS, vals)}, **s})
            disp[f"({layer},{d})"] = f"{s['mean']:.2f} ± {s['std']:.2f}" if m != "base" else f"{s['mean']:.2f}"
        f1_show.append(disp)

    test_rows, test_show = [], []
    for a, b in H11_TESTS:
        for layer, d in H11_CELLS:
            disp = {"comparison": f"{LABEL[a]} - {LABEL[b]}", "cell": f"({layer},{d})"}
            passes = 0
            for t in SEEDS:
                bt = r.boot(a, b, t)
                row = bt[(bt["layer"] == str(layer)) & (bt["dim"] == d)].iloc[0]
                ok = bool(row["ci_low"] > 0)
                passes += ok
                test_rows.append({"comparison": f"{a}-{b}", "layer": layer, "dim": d, "seed": SEEDS[t],
                                  "diff": row["diff"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                                  "passes": ok})
                disp[f"seed {SEEDS[t]}"] = ci(row["diff"], row["ci_low"], row["ci_high"])
            disp["seeds passing"] = f"{passes}/3"
            test_show.append(disp)
    tests = pd.DataFrame(test_rows)
    per_seed = tests.groupby("seed")["passes"].all()
    n_ok = int(per_seed.sum())
    verdict = "SUPPORTED" if n_ok == 3 else "PARTIALLY SUPPORTED" if n_ok == 2 else "NOT SUPPORTED"
    seeds_ok = ", ".join(str(s) for s, ok in per_seed.items() if ok) or "none"
    return (pd.DataFrame(f1_rows), pd.DataFrame(f1_show), tests, pd.DataFrame(test_show),
            f"**{verdict}**: all six comparisons pass in {n_ok} of 3 seeds (passing seeds: {seeds_ok}).")


def table_sts(r: Results):
    rows, show = [], []
    cells = PIC + [("pooler", d) for d in DIMS]
    for m in ("base",) + MODELS:
        disp = {"model": LABEL[m]}
        for layer, d in cells:
            vals = [r.value(r.csv(f"{name(m, t)}_sts_clean.csv"), layer, d, "spearman") * 100 for t in SEEDS]
            s = summarise(vals)
            rows.append({"model": m, "layer": layer, "dim": d,
                         **{f"seed{SEEDS[t]}": v for t, v in zip(SEEDS, vals)}, **s})
            key = f"({layer},{d})" if layer != "pooler" else f"P{d}"
            disp[key] = f"{s['mean']:.2f} ± {s['std']:.2f}" if m != "base" else f"{s['mean']:.2f}"
        show.append(disp)

    boot_rows, boot_show = [], []
    for m in MODELS:
        for layer, d in [(6, 768)] + [("pooler", d) for d in DIMS]:
            disp = {"model": LABEL[m], "cell": f"({layer},{d})" if layer != "pooler" else f"pooler {d}"}
            for t in SEEDS:
                b = r.csv(f"bootstrap_sts_{name(m, t)}_vs_base_clean.csv")
                row = b[(b["layer"] == str(layer)) & (b["dim"] == d)].iloc[0]
                boot_rows.append({"model": m, "layer": layer, "dim": d, "seed": SEEDS[t], "diff": row["diff"],
                                  "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                                  "excludes_zero": bool(row["excludes_zero"])})
                disp[f"seed {SEEDS[t]}"] = ci(row["diff"], row["ci_low"], row["ci_high"])
            boot_show.append(disp)
    return pd.DataFrame(rows), pd.DataFrame(show), pd.DataFrame(boot_rows), pd.DataFrame(boot_show)


def table_interaction(r: Results):
    rows, show = [], []
    for layer, d in PIC + [("pooler", d) for d in DIMS]:
        disp = {"cell": f"({layer},{d})" if layer != "pooler" else f"pooler {d}"}
        vals = []
        for t in SEEDS:
            it = r.csv(f"bootstrap_interaction_{t}.csv")
            row = it[(it["layer"] == str(layer)) & (it["dim"] == d)].iloc[0]
            vals.append(row["interaction"])
            rows.append({"layer": layer, "dim": d, "seed": SEEDS[t], "interaction": row["interaction"],
                         "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                         "excludes_zero": bool(row["excludes_zero"])})
            disp[f"seed {SEEDS[t]}"] = ci(row["interaction"], row["ci_low"], row["ci_high"])
        disp["across seeds"] = cell(summarise(vals))
        show.append(disp)
    table = pd.DataFrame(rows)
    early = table[(table["layer"] == 1) & (table["dim"] == 16)]
    holds = bool(((early["ci_low"] > 0)).all())
    detail = "; ".join(f"seed {int(x.seed)}: {ci(x.interaction, x.ci_low, x.ci_high)}" for x in early.itertuples())
    verdict = f"**{'HOLDS' if holds else 'DOES NOT HOLD'}** at (1,16): {detail}."
    return table, pd.DataFrame(show), verdict


def table_finding1(r: Results):
    rows, show = [], []
    flagged = {t: [] for t in SEEDS}
    for m in MODELS:
        for d in DIMS:
            disp = {"model": LABEL[m], "dim": d}
            for t in SEEDS:
                b = r.boot(m, "base", t)
                row = b[(b["layer"] == "pooler") & (b["dim"] == d)].iloc[0]
                # Stored as model - base; report base - model.
                diff, low, high = -row["diff"], -row["ci_high"], -row["ci_low"]
                rows.append({"probe": "default", "model": m, "dim": d, "seed": SEEDS[t], "size": 26732,
                             "base_minus_model": diff, "ci_low": low, "ci_high": high,
                             "model_significantly_above": bool(high < 0), "base_significantly_above": bool(low > 0)})
                if high < 0:
                    flagged[t].append(f"default {m} {d}D")
                disp[f"default s{SEEDS[t]}"] = ci(diff, low, high)
            show.append(disp)

    tuned_show = []
    for t in SEEDS:
        tb = r.csv(f"probe_tuned_{t}_bootstrap.csv")
        ts = r.csv(f"probe_tuned_{t}_summary.csv")
        for m in MODELS:
            for d in (16, 64, 768):
                row = tb[(tb["model"] == name(m, t)) & (tb["dim"] == d)].iloc[0]
                diff, low, high = -row["diff"], -row["ci_high"], -row["ci_low"]
                rows.append({"probe": "tuned", "model": m, "dim": d, "seed": SEEDS[t], "size": 26732,
                             "base_minus_model": diff, "ci_low": low, "ci_high": high,
                             "model_significantly_above": bool(high < 0), "base_significantly_above": bool(low > 0)})
                if high < 0:
                    flagged[t].append(f"tuned full {m} {d}D")
                for size in (250, 1000, 3000):
                    s = ts[(ts["model"] == name(m, t)) & (ts["size"] == size) & (ts["dim"] == d)].iloc[0]
                    above = bool(s["diff_min"] > 0)
                    rows.append({"probe": "tuned", "model": m, "dim": d, "seed": SEEDS[t], "size": size,
                                 "base_minus_model": -s["diff_mean"], "ci_low": -s["diff_max"],
                                 "ci_high": -s["diff_min"], "model_significantly_above": above,
                                 "base_significantly_above": bool(s["diff_max"] < 0)})
                    if above:
                        flagged[t].append(f"tuned {size} rows {m} {d}D")
    table = pd.DataFrame(rows)
    for (probe, size), g in table.groupby(["probe", "size"]):
        disp = {"probe": probe, "probe rows": size}
        for t in SEEDS:
            gs = g[g["seed"] == SEEDS[t]]
            disp[f"seed {SEEDS[t]}: base above / model above / cells"] = (
                f"{int(gs['base_significantly_above'].sum())} / {int(gs['model_significantly_above'].sum())} / {len(gs)}")
        tuned_show.append(disp)

    n_seeds = sum(1 for t in SEEDS if flagged[t])
    verdict = "HOLDS" if n_seeds == 0 else "WEAKENED" if n_seeds == 1 else "OVERTURNED"
    detail = "; ".join(f"seed {SEEDS[t]}: {', '.join(flagged[t]) if flagged[t] else 'none'}" for t in SEEDS)
    return table, pd.DataFrame(show), pd.DataFrame(tuned_show), f"**{verdict}**. Cells where a trained model is significantly above base: {detail}."


def table_knn_selection(r: Results):
    rows = []
    for t in SEEDS:
        k = r.csv(f"probe_knn_{t}.csv")
        k = k[k["metric"] != "l2norm_euc"]
        for (metric, kk, d), g in k.groupby(["metric", "k", "dim"]):
            base = float(g[g["model"] == "base"]["f1"].iloc[0])
            for m in MODELS:
                rows.append({"seed": SEEDS[t], "metric": metric, "k": kk, "dim": d, "model": m,
                             "base_minus_model": base - float(g[g["model"] == name(m, t)]["f1"].iloc[0])})
    knn = pd.DataFrame(rows)
    knn_show = []
    for (metric, kk, d), g in knn.groupby(["metric", "k", "dim"]):
        disp = {"metric": metric, "k": kk, "dim": d}
        for m in MODELS:
            disp[LABEL[m]] = cell(summarise(g[g["model"] == m]["base_minus_model"].tolist()))
        knn_show.append(disp)

    sel_rows, sel_show = [], []
    for t in SEEDS:
        s = r.csv(f"selection_probe_{t}.csv").set_index("model")["val_macro_f1"]
        sel = json.load(open(r.runs / f"seed{SEEDS[t]}" / "tweeteval_results.json"))["epoch_selection"]
        for m in MODELS:
            curve = {c["epoch"]: c["val_macro_f1"] for c in sel[m]["validation_curve"]}
            sel_rows.append({"seed": SEEDS[t], "model": m, "base": s["base"], "best_pt": s[name(m, t)],
                             "epoch1": curve[1], "epoch1_minus_base": curve[1] - s["base"]})
    sel_df = pd.DataFrame(sel_rows)
    for m in MODELS:
        g = sel_df[sel_df["model"] == m]
        sel_show.append({"model": LABEL[m], **{f"seed {s} epoch 1": f"{v:.2f}" for s, v in zip(g["seed"], g["epoch1"])},
                         "epoch 1 minus base (63.16)": cell(summarise(g["epoch1_minus_base"].tolist()))})
    return knn, pd.DataFrame(knn_show), sel_df, pd.DataFrame(sel_show)


def table_subsets(r: Results):
    rows, show = [], []
    for m in ("base",) + MODELS:
        disp = {"model": LABEL[m]}
        for d in (16, 32, 64, 128):
            vals = [r.value(r.csv(f"{name(m, t)}_subsets_summary.csv"), "pooler", d, "prefix_minus_random") * 100
                    for t in SEEDS]
            s = summarise(vals)
            rows.append({"model": m, "dim": d, **{f"seed{SEEDS[t]}": v for t, v in zip(SEEDS, vals)}, **s})
            disp[str(d)] = cell(s) if m != "base" else f"{s['mean']:.2f}"
        show.append(disp)
    return pd.DataFrame(rows), pd.DataFrame(show)


def table_epochs(r: Results):
    rows = []
    for t in SEEDS:
        sel = json.load(open(r.runs / f"seed{SEEDS[t]}" / "tweeteval_results.json"))["epoch_selection"]
        for m in MODELS:
            rows.append({"seed": SEEDS[t], "model": m, "selected_epoch": sel[m]["selected_epoch"]})
    table = pd.DataFrame(rows)
    show = table.pivot(index="model", columns="seed", values="selected_epoch")[[42, 1, 2]].reset_index()
    show["model"] = show["model"].map(LABEL)
    show.columns = ["model"] + [f"seed {c}" for c in show.columns[1:]]
    return table, show


def table_finding9(r: Results):
    rows = []
    for t in SEEDS:
        w = r.csv(f"weight_drift_{t}.csv")
        fwd = r.csv(f"pooler_swap_{t}.csv")
        rev = r.csv(f"pooler_swap_reverse_{t}.csv")
        base_f1 = fwd[fwd["pooler_weights"] == "base"].set_index("dim")["f1"]
        for m in MODELS:
            n = name(m, t)
            wm = w[w["model"] == n].set_index("group")
            f = fwd[fwd["pooler_weights"] == n].set_index("dim")
            own = rev[(rev["model"] == n) & (rev["readout"] == "own")].set_index("dim")
            bp = rev[(rev["model"] == n) & (rev["readout"] == "base_pooler")].set_index("dim")
            rec = {"seed": SEEDS[t], "model": m,
                   "encoder_change_pct": wm.loc["encoder", "relative_change"] * 100,
                   "layer1_change_pct": wm.loc["layer1", "relative_change"] * 100,
                   "layer6_change_pct": wm.loc["layer6", "relative_change"] * 100,
                   "pooler_change_pct": wm.loc["pooler", "relative_change"] * 100,
                   "fwd_sd_ratio": f["sd_ratio_to_base"].iloc[0],
                   "rev_sd_ratio_base_pooler": bp["sd_ratio_to_base"].iloc[0],
                   "own_sd_ratio": own["sd_ratio_to_base"].iloc[0]}
            for d in (16, 64, 768):
                rec[f"fwd_f1_minus_base_{d}"] = f.loc[d, "f1"] - base_f1.loc[d]
                rec[f"rev_f1_minus_own_{d}"] = bp.loc[d, "f1"] - own.loc[d, "f1"]
            rows.append(rec)
    table = pd.DataFrame(rows)
    show = []
    for m in MODELS:
        g = table[table["model"] == m]
        show.append({"model": LABEL[m],
                     "encoder change %": cell(summarise(g["encoder_change_pct"].tolist())),
                     "pooler change %": cell(summarise(g["pooler_change_pct"].tolist())),
                     "fwd: trained pooler on base h6, SD ratio": cell(summarise(g["fwd_sd_ratio"].tolist()), 3),
                     "fwd: F1 minus base, 16/64/768": " / ".join(
                         f"{summarise(g[f'fwd_f1_minus_base_{d}'].tolist())['mean']:+.2f}" for d in (16, 64, 768)),
                     "rev: base pooler on trained h6, SD ratio": cell(summarise(g["rev_sd_ratio_base_pooler"].tolist()), 3),
                     "rev: F1 minus own, 16/64/768": " / ".join(
                         f"{summarise(g[f'rev_f1_minus_own_{d}'].tolist())['mean']:+.2f}" for d in (16, 64, 768))})
    return table, pd.DataFrame(show)


def table_readme(r: Results):
    readme = json.load(open(REPO_ROOT / "layerwise_probe" / "expected_readme.json"))["macro_f1"]
    rows, show = [], []
    for m in MODELS:
        disp = {"model": LABEL[m]}
        for d in DIMS:
            vals = [r.value(r.grid(m, t), "pooler", d, "f1") for t in SEEDS]
            s = summarise(vals)
            gap = s["mean"] - readme[m][str(d)]
            rows.append({"model": m, "dim": d, "readme": readme[m][str(d)], "seed_mean": s["mean"],
                         "seed_std": s["std"], "seed_min": s["min"], "seed_max": s["max"], "gap": gap,
                         "gap_over_std": gap / s["std"] if s["std"] > 0 else np.nan,
                         "readme_below_all_seeds": readme[m][str(d)] < s["min"]})
            disp[str(d)] = f"{gap:+.2f} (std {s['std']:.2f})"
        show.append(disp)
    return pd.DataFrame(rows), pd.DataFrame(show)


# ---------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-seed summary for seeds 42, 1 and 2.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "results")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "layerwise_probe" / "runs")
    parser.add_argument("--out-md", type=Path, default=REPO_ROOT / "layerwise_probe" / "CROSS_SEED.md")
    args = parser.parse_args(argv)
    r = Results(args.results_dir, args.runs_dir)

    def save(frame: pd.DataFrame, stem: str) -> str:
        frame.to_csv(args.results_dir / f"cross_seed_{stem}.csv", index=False)
        return f"`results/cross_seed_{stem}.csv`"

    t1, t1s = table_pooler_f1(r)
    h11_f1, h11_f1s, h11_t, h11_ts, h11_v = table_h11(r)
    sts, stss, sts_b, sts_bs = table_sts(r)
    inter, inters, h16_v = table_interaction(r)
    f1, f1s, f1c, f1_v = table_finding1(r)
    knn, knns, sel, sels = table_knn_selection(r)
    sub, subs = table_subsets(r)
    ep, eps = table_epochs(r)
    f9, f9s = table_finding9(r)
    rd, rds = table_readme(r)

    sections = [
        ("1. Sentiment pooler F1 per model per width (README-style)",
         "Mean ± sample std over seeds 42, 1, 2 (macro-F1, default probe, full probe split). Base has no seed.",
         [t1s], [save(t1, "pooler_f1")]),
        ("2. H11: sentiment F1 at the six PIC checkpoints, and the decision-rule comparisons",
         f"Decision rule verdict: {h11_v} A comparison passes when its paired bootstrap CI is entirely above zero. "
         "Non-convergence: the (5,512) fit hit max_iter for MRL+SIA in all three seeds and for MRL in seed 2.",
         [h11_f1s, h11_ts], [save(h11_f1, "h11_f1"), save(h11_t, "h11_tests")]),
        ("3. STS-B clean Spearman at the six PIC checkpoints and at the pooler",
         "Spearman x 100 on the 1,261 clean pairs; mean ± std. The second table is the paired bootstrap of model "
         "minus base (1,000 resamples of pairs, seed 0).",
         [stss, sts_bs], [save(sts, "sts_clean"), save(sts_b, "sts_bootstrap")]),
        ("4. H16: SIA x PIC interaction (+MIPIC - MRL+SIA - MRL+PIC + MRL)",
         f"Decision rule verdict: {h16_v} Across-seed column: mean ± std [min, max] and how many seeds share the "
         "sign of the mean.", [inters], [save(inter, "interaction")]),
        ("5. Finding 1: base minus each model at the pooler",
         f"Decision rule verdict: {f1_v} Default probe: paired bootstrap per seed. Tuned probe at full size: paired "
         "bootstrap; at 250, 1,000 and 3,000 rows: 'above' means ahead on all 5 subsets. Second table: counts of "
         "cells where base is significantly above, where the model is, and the total.",
         [f1s, f1c], [save(f1, "finding1")]),
        ("5b. Finding 1 checks 7 and 8: kNN and the selection probe",
         "kNN: base minus model, macro-F1, mean ± std [min, max] and sign agreement across seeds (no CIs). "
         "Selection probe: base scores 63.16 in every seed (it has no training seed).",
         [knns, sels], [save(knn, "knn"), save(sel, "selection")]),
        ("6. Subset probe: pooler prefix minus mean random subset",
         "Macro-F1 points; mean ± std [min, max] and sign agreement.", [subs], [save(sub, "subsets")]),
        ("7. Selected epoch per model per seed (Finding 8)", "", [eps], [save(ep, "epochs")]),
        ("8. Finding 9: weight drift and both pooler swaps",
         "fwd = trained pooler on base h6 (vs base's pooler); rev = base pooler on trained h6 (vs the model's own "
         "pooler output). SD ratios are relative to base's pooler features. F1 differences are seed means.",
         [f9s], [save(f9, "finding9")]),
        ("9. Finding 4: local seed mean minus README, per model per width",
         "Each cell: seed mean minus the README value, with the across-seed std in brackets.",
         [rds], [save(rd, "readme")]),
    ]

    lines = ["# Cross-seed summary (seeds 42, 1 and 2)", "",
             "Generated by `layerwise_probe/aggregate_seeds.py` from the per-seed files in `layerwise_probe/results/`. "
             "Across-seed cells show mean ± sample std [min, max] and, where given, how many of the three seeds "
             "share the sign of the mean. With three seeds, no seed-level p-values or t-intervals are reported; "
             "per-seed significance comes from paired bootstraps over test rows or STS pairs (* = 95% CI excludes 0). "
             "KL-on runs exist for seed 42 only and are not included.", ""]
    for title, note, frames, files in sections:
        lines += [f"## {title}", ""]
        if note:
            lines += [note, ""]
        for frame in frames:
            lines += [md_table(frame), ""]
        lines += ["Files: " + ", ".join(files), ""]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"H11: {h11_v}\nFinding 1: {f1_v}\nH16: {h16_v}")
    print(f"Saved {args.out_md} and results/cross_seed_*.csv")


if __name__ == "__main__":
    main()
