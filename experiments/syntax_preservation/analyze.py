"""Paired bootstrap, cross-seed aggregation and the syntax-preservation decision rules.

Reads every experiments/syntax_preservation/results/<name>_probe.csv and <name>_preds.npz (step 3), where
<name> is "base" or "<arm>_s<seed>" with arm in mrl, mrl_pic, mipic, mrl_sia.
Scores are in points: POS accuracy x 100, sentiment macro-F1 x 100 (mean over
the 3 probe seeds).

Bootstrap (1,000 paired resamples, seed 0):
    POS        resamples UD test SENTENCES (2,077), not words: words of one
               sentence are not independent, and resampling words would make
               the intervals too narrow.
    sentiment  resamples test tweets (3,432); in each resample, macro-F1 is
               averaged over the 3 probe seeds, as the point estimate is.
The same resample indices are used for every comparison.

Writes, in experiments/syntax_preservation/results/:
    summary_cells.csv   per arm, task, cell: score in each seed, mean, sample SD
    bootstrap.csv       per seed and cell: model minus reference with 95% CI,
                        for each trained arm vs MRL of the same seed, and every
                        model vs base
    decisions.csv       every rule and check, with its numbers and verdict
and experiments/syntax_preservation/RESULTS.md, the same in readable tables (generated; do not edit by hand).

Example, from the repository root:
    python experiments/syntax_preservation/analyze.py
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.syntax_preservation import probe  # noqa: E402

ARMS = ("mrl", "mrl_pic", "mipic", "mrl_sia")
ARM_LABEL = {"base": "Base", "mrl": "MRL", "mrl_pic": "MRL+PIC", "mipic": "MIPIC", "mrl_sia": "MRL+SIA"}
SEEDS = (42, 1, 2)
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
RULE_B_LAYERS = (1, 2, 3)
K_SIGMA = 2.0
FLOOR = 0.5  # points
TASK_METRIC = {"pos": "accuracy", "sentiment": "macro-F1"}
DEFAULT_RESULTS_DIR = probe.DEFAULT_OUT_DIR
DEFAULT_REPORT = REPO_ROOT / "experiments" / "syntax_preservation" / "RESULTS.md"
CELLS = probe.probe_cells()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def parse_name(name: str) -> Tuple[str, Optional[int]]:
    if name == "base":
        return "base", None
    match = re.fullmatch(r"(mrl|mrl_pic|mipic|mrl_sia)_s(\d+)", name)
    if not match:
        raise ValueError(f"unrecognised result name {name!r}")
    return match.group(1), int(match.group(2))


class Model:
    """One model's probe results and test predictions."""

    def __init__(self, results_dir: Path, name: str):
        self.name = name
        self.arm, self.seed = parse_name(name)
        self.table = pd.read_csv(results_dir / f"{name}_probe.csv")
        with np.load(results_dir / f"{name}_preds.npz") as stored:
            self.preds = dict(stored)

    def score(self, task: str, cell: probe.Cell) -> float:
        rows = self.table[(self.table.task == task) & (self.table.layer == cell.layer)
                          & (self.table.coords == cell.coords)]
        if rows.empty:
            raise KeyError(f"{self.name}: no {task} result for {cell.key}")
        column = "test_accuracy" if task == "pos" else "test_macro_f1"
        return float(rows[column].mean()) * 100

    def predictions(self, task: str, cell: probe.Cell) -> np.ndarray:
        """(probe seeds, test rows): one row for POS, three for sentiment."""
        keys = sorted(k for k in self.preds if k.startswith(f"{task}_{cell.key}_s"))
        return np.stack([self.preds[k] for k in keys])


def load_models(results_dir: Path) -> Dict[str, Model]:
    names = sorted(p.name[:-len("_probe.csv")] for p in results_dir.glob("*_probe.csv"))
    models = {}
    for name in names:
        try:
            parse_name(name)
        except ValueError:
            continue
        if (results_dir / f"{name}_preds.npz").exists():
            models[name] = Model(results_dir, name)
    return models


def pos_sentence_index(n_words: int) -> np.ndarray:
    """Sentence index of each UD test word, rebuilt from the UD files (no word is dropped)."""
    from experiments.syntax_preservation import ud_data

    owner = np.concatenate([np.full(len(s.words), i) for i, s in enumerate(ud_data.load_split("test"))])
    if len(owner) != n_words:
        raise ValueError(f"UD test has {len(owner)} words but the predictions have {n_words}")
    return owner


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def macro_f1_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Macro-F1 along the last axis (sklearn's definition: 0 for a class with 2TP+FP+FN = 0)."""
    scores = []
    for c in classes:
        true_c, pred_c = y_true == c, y_pred == c
        tp = np.sum(true_c & pred_c, axis=-1)
        denom = 2 * tp + np.sum(~true_c & pred_c, axis=-1) + np.sum(true_c & ~pred_c, axis=-1)
        scores.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return np.mean(scores, axis=0)


class Bootstrap:
    """Paired resampling shared by every comparison."""

    def __init__(self, pos_labels: np.ndarray, pos_owner: np.ndarray, sentiment_labels: np.ndarray,
                 n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED):
        rng = np.random.default_rng(seed)
        self.pos_labels, self.sentiment_labels = pos_labels, sentiment_labels
        self.n_sentences = int(pos_owner.max()) + 1
        self.pos_owner = pos_owner
        self.words_per_sentence = np.bincount(pos_owner, minlength=self.n_sentences)
        self.sentence_idx = rng.integers(0, self.n_sentences, size=(n_resamples, self.n_sentences))
        self.row_idx = rng.integers(0, len(sentiment_labels), size=(n_resamples, len(sentiment_labels)))
        self.classes = np.unique(sentiment_labels)

    def pos(self, pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Accuracy (points): observed, and per resample."""
        correct = np.bincount(self.pos_owner, weights=(pred[0] == self.pos_labels), minlength=self.n_sentences)
        observed = correct.sum() / self.words_per_sentence.sum()
        boot = correct[self.sentence_idx].sum(axis=1) / self.words_per_sentence[self.sentence_idx].sum(axis=1)
        return observed * 100, boot * 100

    def sentiment(self, pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Macro-F1 averaged over probe seeds (points): observed, and per resample."""
        y = self.sentiment_labels
        observed = np.mean([macro_f1_rows(y, p, self.classes) for p in pred])
        y_boot = y[self.row_idx]
        boot = np.mean([macro_f1_rows(y_boot, p[self.row_idx], self.classes) for p in pred], axis=0)
        return float(observed) * 100, boot * 100

    def score(self, task: str, pred: np.ndarray) -> Tuple[float, np.ndarray]:
        return self.pos(pred) if task == "pos" else self.sentiment(pred)


def compare(boot: Bootstrap, a: Model, b: Model, task: str, cell: probe.Cell) -> Dict[str, object]:
    score_a, boot_a = boot.score(task, a.predictions(task, cell))
    score_b, boot_b = boot.score(task, b.predictions(task, cell))
    low, high = np.percentile(boot_a - boot_b, [2.5, 97.5])
    return {"a": a.name, "b": b.name, "seed": a.seed if a.seed is not None else b.seed, "task": task,
            "layer": cell.layer, "coords": cell.coords, "start": cell.start, "stop": cell.stop,
            "score_a": score_a, "score_b": score_b, "diff": score_a - score_b,
            "ci_low": float(low), "ci_high": float(high), "excludes_zero": bool(low > 0 or high < 0)}


def peak_distribution(boot: Bootstrap, model: Model, task: str) -> Tuple[int, np.ndarray]:
    """Observed peak layer of the full-width profile, and the peak layer in each resample.

    The same resample is used for every layer, so the layers are compared paired.
    Ties go to the earlier layer (np.argmax).
    """
    cells = [c for c in CELLS if c.coords == "full"]
    observed, boots = zip(*(boot.score(task, model.predictions(task, c)) for c in cells))
    layers = np.array([c.layer for c in cells])
    return int(layers[int(np.argmax(observed))]), layers[np.argmax(np.stack(boots, axis=1), axis=1)]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def seed_noise(diffs: Sequence[float], cis: Sequence[Tuple[float, float]], reference_scores: Sequence[float]) -> Dict[str, object]:
    """The pre-registered seed-noise rule: T = max(2 sigma, 0.5), sigma = SD of the reference over seeds."""
    n = len(diffs)
    sigma = float(np.std(reference_scores, ddof=1)) if n >= 2 else float("nan")
    threshold = max(K_SIGMA * sigma, FLOOR) if n >= 2 else float("nan")
    mean = float(np.mean(diffs)) if n else float("nan")
    all_below = n > 0 and all(high < 0 for _, high in cis)
    all_above = n > 0 and all(low > 0 for low, _ in cis)
    if n < 2:
        verdict = "pending"
    elif mean < -threshold and all_below:
        verdict = "damage"
    elif abs(mean) <= threshold:
        verdict = "within noise"
    else:
        verdict = "inconclusive"
    effect = n >= 2 and abs(mean) > threshold and (all_below or all_above)
    return {"n_seeds": n, "sigma": sigma, "threshold": threshold, "mean_diff": mean,
            "all_ci_below_zero": all_below, "all_ci_above_zero": all_above,
            "verdict": verdict, "measurable_effect": bool(effect)}


def by_seed(models: Dict[str, Model], arm: str) -> Dict[int, Model]:
    return {m.seed: m for m in models.values() if m.arm == arm}


def full_cell(layer: int) -> probe.Cell:
    return probe.Cell(layer, "full", 0, probe.HIDDEN)


def checkpoint_cells(layer: int, dim: int) -> Tuple[probe.Cell, probe.Cell]:
    return probe.Cell(layer, "prefix", 0, dim), probe.Cell(layer, "suffix", dim, probe.HIDDEN)


def analyze(models: Dict[str, Model], boot: Bootstrap) -> Dict[str, pd.DataFrame]:
    arms = defaultdict(dict)
    for model in models.values():
        arms[model.arm][model.seed] = model

    # Cross-seed summary of every cell.
    summary = []
    for arm, seeds in arms.items():
        for task in probe.TASKS:
            for cell in CELLS:
                scores = {s: m.score(task, cell) for s, m in seeds.items()}
                values = list(scores.values())
                summary.append({"arm": arm, "task": task, "layer": cell.layer, "coords": cell.coords,
                                "start": cell.start, "stop": cell.stop, "n_seeds": len(values),
                                "mean": float(np.mean(values)),
                                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                                **{f"seed_{s}" if s is not None else "score": v for s, v in scores.items()}})
    summary = pd.DataFrame(summary)

    # Paired bootstraps.
    comparisons = []
    mrl = by_seed(models, "mrl")
    base = models.get("base")
    for model in models.values():
        if model.arm == "base":
            continue
        for task in probe.TASKS:
            for cell in CELLS:
                if model.arm != "mrl" and model.seed in mrl:
                    comparisons.append(compare(boot, model, mrl[model.seed], task, cell))
                if base is not None:
                    comparisons.append(compare(boot, model, base, task, cell))
    # Explicit columns, so the CSV keeps its header even before any comparison exists (base alone).
    bootstrap = pd.DataFrame(comparisons, columns=[
        "a", "b", "seed", "task", "layer", "coords", "start", "stop",
        "score_a", "score_b", "diff", "ci_low", "ci_high", "excludes_zero"])

    def lookup(a: str, b: str, task: str, cell: probe.Cell) -> pd.Series:
        rows = bootstrap[(bootstrap.a == a) & (bootstrap.b == b) & (bootstrap.task == task)
                         & (bootstrap.layer == cell.layer) & (bootstrap.coords == cell.coords)]
        return rows.iloc[0]

    decisions = []

    # Rule (a): peak layers in base (the rule), and in every model (descriptive).
    for model in models.values():
        pos_peak, pos_boot = peak_distribution(boot, model, "pos")
        sent_peak, sent_boot = peak_distribution(boot, model, "sentiment")
        # The two tasks use different test sets, so their resamples are independent.
        p_earlier = float(np.mean(pos_boot[:, None] < sent_boot[None, :]))
        is_rule = model.arm == "base"
        decisions.append({
            "rule": "a" if is_rule else "a (descriptive)", "model": model.name, "arm": model.arm, "seed": model.seed,
            "pos_peak_layer": pos_peak, "sentiment_peak_layer": sent_peak,
            "p_pos_peak_earlier": p_earlier,
            "pos_peak_freq": _freq(pos_boot), "sentiment_peak_freq": _freq(sent_boot),
            "verdict": ("supported" if pos_peak < sent_peak else "not supported") if is_rule
            else ("POS earlier" if pos_peak < sent_peak else "POS not earlier"),
        })

    mipic = by_seed(models, "mipic")
    mrl_pic = by_seed(models, "mrl_pic")
    shared = sorted(set(mipic) & set(mrl))

    # Rule (b) and falsification 1: MIPIC vs MRL, POS accuracy, full width, layers 1-3.
    layer_verdicts = []
    for layer in RULE_B_LAYERS:
        cell = full_cell(layer)
        rows = [lookup(mipic[s].name, mrl[s].name, "pos", cell) for s in shared]
        result = seed_noise([r["diff"] for r in rows], [(r["ci_low"], r["ci_high"]) for r in rows],
                            [mrl[s].score("pos", cell) for s in shared])
        layer_verdicts.append(result["verdict"])
        decisions.append({"rule": "b", "layer": layer, "coords": "full", "task": "pos",
                          "seeds": ",".join(map(str, shared)),
                          "per_seed": _per_seed(shared, rows), **result})
    if not shared or "pending" in layer_verdicts:
        overall_b = "pending"
    elif "damage" in layer_verdicts:
        overall_b = "falsified (damage)"
    elif all(v == "within noise" for v in layer_verdicts):
        overall_b = "supported"
    else:
        overall_b = "inconclusive"
    decisions.append({"rule": "b (overall)", "verdict": overall_b,
                      "provisional": len(shared) < len(SEEDS), "seeds": ",".join(map(str, shared))})

    # Rule (c): at each PIC checkpoint, in MIPIC POS(suffix) >= POS(prefix) in every seed,
    # and MIPIC sentiment(prefix) above MRL's with the CI above zero in every seed.
    supported_at = 0
    for layer, dim in probe.PIC_CHECKPOINTS:
        prefix, suffix = checkpoint_cells(layer, dim)
        pos_gaps = [mipic[s].score("pos", suffix) - mipic[s].score("pos", prefix) for s in shared]
        rows = [lookup(mipic[s].name, mrl[s].name, "sentiment", prefix) for s in shared]
        pos_ok = bool(shared) and all(g >= 0 for g in pos_gaps)
        sent_ok = bool(shared) and all(r["ci_low"] > 0 for r in rows)
        supported_at += pos_ok and sent_ok
        decisions.append({
            "rule": "c", "layer": layer, "dim": dim, "seeds": ",".join(map(str, shared)),
            "pos_suffix_minus_prefix": _per_seed_values(shared, pos_gaps),
            "pos_part": pos_ok,
            "sentiment_prefix_mipic_minus_mrl": _per_seed(shared, rows),
            "sentiment_part": sent_ok,
            "verdict": "pending" if not shared else ("holds" if pos_ok and sent_ok else "fails"),
        })
    n_checkpoints = len(probe.PIC_CHECKPOINTS)
    overall_c = ("pending" if not shared else "supported" if supported_at == n_checkpoints
                 else f"supported at {supported_at} of {n_checkpoints} checkpoints")
    decisions.append({"rule": "c (overall)", "verdict": overall_c,
                      "provisional": len(shared) < len(SEEDS), "seeds": ",".join(map(str, shared))})

    # Falsification 2: does PIC have a measurable effect? MRL+PIC vs MRL, full width, layers 1-3.
    shared_pic = sorted(set(mrl_pic) & set(mrl))
    effects = []
    for task in probe.TASKS:
        for layer in RULE_B_LAYERS:
            cell = full_cell(layer)
            rows = [lookup(mrl_pic[s].name, mrl[s].name, task, cell) for s in shared_pic]
            result = seed_noise([r["diff"] for r in rows], [(r["ci_low"], r["ci_high"]) for r in rows],
                                [mrl[s].score(task, cell) for s in shared_pic])
            effects.append(result["measurable_effect"])
            decisions.append({"rule": "pic effect", "task": task, "layer": layer, "coords": "full",
                              "seeds": ",".join(map(str, shared_pic)), "per_seed": _per_seed(shared_pic, rows),
                              **{k: v for k, v in result.items() if k != "verdict"},
                              "verdict": "pending" if len(shared_pic) < 2 else
                              ("effect" if result["measurable_effect"] else "no effect")})
    overall_pic = ("pending" if len(shared_pic) < 2 else
                   "PIC has a measurable effect" if any(effects) else "falsified (no measurable effect)")
    decisions.append({"rule": "pic effect (overall)", "verdict": overall_pic,
                      "provisional": len(shared_pic) < len(SEEDS), "seeds": ",".join(map(str, shared_pic))})

    return {"summary": summary, "bootstrap": bootstrap, "decisions": pd.DataFrame(decisions)}


def _freq(layers: np.ndarray) -> str:
    values, counts = np.unique(layers, return_counts=True)
    return "; ".join(f"L{v}: {c / len(layers):.1%}" for v, c in zip(values, counts))


def _per_seed(seeds: Sequence[int], rows: Sequence[pd.Series]) -> str:
    return "; ".join(f"s{s}: {r['diff']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]" for s, r in zip(seeds, rows))


def _per_seed_values(seeds: Sequence[int], values: Sequence[float]) -> str:
    return "; ".join(f"s{s}: {v:+.2f}" for s, v in zip(seeds, values))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.2f}" if np.isnan(sd) else f"{mean:.2f} ± {sd:.2f}"


def _num(value: float, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _arm_order(arms: Sequence[str]) -> List[str]:
    return [a for a in ("base",) + ARMS if a in arms]


def render_report(models: Dict[str, Model], out: Dict[str, pd.DataFrame]) -> str:
    summary, decisions = out["summary"], out["decisions"]
    arms = _arm_order(summary["arm"].unique())
    seeds = sorted({m.seed for m in models.values() if m.seed is not None}, key=SEEDS.index)
    lines = ["# Syntax preservation: results", "",
             "Generated by `python experiments/syntax_preservation/analyze.py` from the files in `results/`; do not edit by hand. "
             "The test design, decision rules and their reasons are in [README.md](README.md).", "",
             f"Models: {', '.join(sorted(models))}. Training seeds found: {', '.join(map(str, seeds)) or 'none'}. "
             "Values are mean ± sample SD over training seeds (base: one model). "
             "POS: accuracy (%) on UD EWT test. Sentiment: macro-F1 (%) on the tweet test set, mean over 3 probe seeds.", ""]
    if len(seeds) < len(SEEDS):
        lines += [f"**Provisional:** {len(seeds)} of {len(SEEDS)} training seeds. The seed-noise rule needs all three.", ""]

    def cell_table(task: str, cells: Sequence[probe.Cell], label) -> List[str]:
        header = "| " + " | ".join(["Cell"] + [ARM_LABEL[a] for a in arms]) + " |"
        rows = [header, "|" + "---|" * (len(arms) + 1)]
        for cell in cells:
            entries = []
            for arm in arms:
                r = summary[(summary.arm == arm) & (summary.task == task) & (summary.layer == cell.layer)
                            & (summary.coords == cell.coords)].iloc[0]
                entries.append(_fmt(r["mean"], r["sd"]))
            rows.append("| " + " | ".join([label(cell)] + entries) + " |")
        return rows

    full = [c for c in CELLS if c.coords == "full"]
    checkpoint = [c for c in CELLS if c.coords != "full"]
    for task in probe.TASKS:
        name = "POS accuracy" if task == "pos" else "Sentiment macro-F1"
        lines += [f"## Layer profiles: {name}", ""] + cell_table(task, full, lambda c: f"Layer {c.layer}") + [""]
    for task in probe.TASKS:
        name = "POS accuracy" if task == "pos" else "Sentiment macro-F1"
        lines += [f"## Prefix and suffix at the PIC checkpoints: {name}", ""]
        lines += cell_table(task, checkpoint, lambda c: f"L{c.layer} {c.coords} [{c.start}:{c.stop}]") + [""]

    rule_a = decisions[decisions.rule.str.startswith("a")]
    lines += ["## Rule (a): POS peaks at a lower layer than sentiment", "",
              "Peak = layer with the highest full-width score. P(POS earlier) = share of bootstrap resamples in "
              "which the POS peak is at a lower layer than the sentiment peak. The rule uses base; the trained "
              "models are shown for comparison.", "",
              "| Model | POS peak | Sentiment peak | P(POS earlier) | POS peak in resamples | Sentiment peak in resamples | Verdict |",
              "|---|---:|---:|---:|---|---|---|"]
    for r in rule_a.itertuples():
        lines.append(f"| {r.model} | {r.pos_peak_layer:.0f} | {r.sentiment_peak_layer:.0f} | {r.p_pos_peak_earlier:.2f} | "
                     f"{r.pos_peak_freq} | {r.sentiment_peak_freq} | {r.verdict} |")
    lines.append("")

    def noise_table(rows: pd.DataFrame, task_col: bool) -> List[str]:
        head = ["Task"] if task_col else []
        out = ["| " + " | ".join(head + ["Layer", "Per seed: diff [95% CI]", "Mean diff", "σ (MRL)", "T", "Verdict"]) + " |",
               "|" + "---|" * (len(head) + 6)]
        for r in rows.itertuples():
            first = [r.task] if task_col else []
            out.append("| " + " | ".join(first + [f"{r.layer:.0f}", r.per_seed or "–", _num(r.mean_diff, True),
                                                  _num(r.sigma), _num(r.threshold), r.verdict]) + " |")
        return out

    rule_b = decisions[decisions.rule == "b"]
    overall_b = decisions[decisions.rule == "b (overall)"].iloc[0]
    lines += ["## Rule (b): MIPIC keeps POS information at layers 1–3", "",
              "MIPIC minus MRL, POS accuracy, all 768 coordinates, paired within each training seed. "
              "T = max(2σ, 0.5) with σ the SD of MRL over seeds. Damage: mean diff < −T and every CI below zero. "
              "Within noise: |mean diff| ≤ T. Otherwise inconclusive.", ""] + noise_table(rule_b, False)
    lines += ["", f"**Rule (b): {overall_b.verdict}.**", ""]

    rule_c = decisions[decisions.rule == "c"]
    overall_c = decisions[decisions.rule == "c (overall)"].iloc[0]
    lines += ["## Rule (c): POS outside the prefix, sentiment gain inside it", "",
              "At each checkpoint, in MIPIC: POS(suffix) − POS(prefix) ≥ 0 in every seed, and MIPIC minus MRL "
              "sentiment(prefix) with the 95% CI above zero in every seed.", "",
              "| Checkpoint | MIPIC POS suffix − prefix | POS part | MIPIC − MRL sentiment prefix [95% CI] | Sentiment part | Verdict |",
              "|---|---|---|---|---|---|"]
    for r in rule_c.itertuples():
        def part(ok) -> str:
            return "–" if r.verdict == "pending" else ("yes" if ok else "no")
        lines.append(f"| ({r.layer:.0f},{r.dim:.0f}) | {r.pos_suffix_minus_prefix or '–'} | {part(r.pos_part)} | "
                     f"{r.sentiment_prefix_mipic_minus_mrl or '–'} | {part(r.sentiment_part)} | {r.verdict} |")
    lines += ["", f"**Rule (c): {overall_c.verdict}.**", ""]

    pic = decisions[decisions.rule == "pic effect"]
    overall_pic = decisions[decisions.rule == "pic effect (overall)"].iloc[0]
    lines += ["## Falsification check: does PIC change the early layers at all?", "",
              "MRL+PIC minus MRL, all 768 coordinates, layers 1–3. An effect needs |mean diff| > T and every CI "
              "excluding zero in the same direction.", ""] + noise_table(pic, True)
    lines += ["", f"**PIC effect: {overall_pic.verdict}.**", ""]

    lines += ["## Summary", "",
              "| Rule | Verdict |", "|---|---|",
              f"| (a) POS peaks before sentiment, base | {rule_a[rule_a.rule == 'a'].verdict.iloc[0] if (rule_a.rule == 'a').any() else 'pending (no base results)'} |",
              f"| (b) MIPIC keeps POS at layers 1–3 | {overall_b.verdict} |",
              f"| (c) POS in the suffix, sentiment gain in the prefix | {overall_c.verdict} |",
              f"| PIC has a measurable effect on early layers | {overall_pic.verdict} |", ""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap, aggregate and apply the syntax-preservation decision rules.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-resamples", type=int, default=None, help=f"Default {N_RESAMPLES}.")
    args = parser.parse_args(argv)

    models = load_models(args.results_dir)
    if not models:
        raise FileNotFoundError(f"no results in {args.results_dir}; run probe.py first")
    any_model = next(iter(models.values()))
    pos_labels = any_model.preds["pos_labels"]
    sentiment_labels = any_model.preds["sentiment_labels"]
    for m in models.values():
        if not (np.array_equal(m.preds["pos_labels"], pos_labels)
                and np.array_equal(m.preds["sentiment_labels"], sentiment_labels)):
            raise ValueError(f"{m.name} was scored on different test labels; paired comparisons are invalid")
    boot = Bootstrap(pos_labels, pos_sentence_index(len(pos_labels)), sentiment_labels,
                     n_resamples=args.n_resamples or N_RESAMPLES)

    out = analyze(models, boot)
    out["summary"].to_csv(args.results_dir / "summary_cells.csv", index=False)
    out["bootstrap"].to_csv(args.results_dir / "bootstrap.csv", index=False)
    out["decisions"].to_csv(args.results_dir / "decisions.csv", index=False)
    report = render_report(models, out)
    args.report.write_text(report, encoding="utf-8")
    print(report[report.index("## Summary"):])
    print(f"Saved summary_cells.csv, bootstrap.csv and decisions.csv to {args.results_dir}, and {args.report}")


if __name__ == "__main__":
    main()
