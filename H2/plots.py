"""Figures for H2, drawn from the tables analyze.py writes.

Reads H2/results/summary_cells.csv, bootstrap.csv and decisions.csv and writes
three PNGs to H2/figures/:

    layer_profiles.png   POS accuracy and sentiment macro-F1 by layer, one line per
                         model (mean over training seeds, whiskers +-1 SD), in two
                         panels: two measures never share an axis. Peak layers are
                         marked (rule a).
    seed_noise.png       Rule (b) and the PIC-effect check: per-seed difference with
                         its 95% bootstrap CI, the +-T noise band in grey, zero line.
    checkpoints.png      Rule (c): POS in the prefix and the suffix at each PIC
                         checkpoint, and MIPIC minus MRL sentiment in the prefix.

Colours follow the model, never its rank: MRL blue, MRL+PIC orange, MIPIC aqua,
MRL+SIA yellow (categorical slots 1-4, validated for colour-vision deficiency on
adjacent pairs), base neutral grey as the untrained reference. Every model also
has its own marker shape and a direct label, so identity never rests on colour
alone; RESULTS.md holds every plotted number as a table.

Example, from the repository root (after analyze.py):
    python H2/plots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from H2 import analyze, probe  # noqa: E402

DEFAULT_FIGURES_DIR = REPO_ROOT / "H2" / "figures"

# Reference palette (light mode): chart chrome and categorical slots 1-4.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BAND = "#f0efec"
COLOR = {"base": MUTED, "mrl": "#2a78d6", "mrl_pic": "#eb6834", "mipic": "#1baf7a", "mrl_sia": "#eda100"}
MARKER = {"base": "s", "mrl": "o", "mrl_pic": "^", "mipic": "D", "mrl_sia": "v"}
LABEL = analyze.ARM_LABEL
SEED_MARKER = {42: "o", 1: "s", 2: "^"}
DPI = 150


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
        "lines.linewidth": 2, "lines.markersize": 5.5,
    })


def arms_present(summary: pd.DataFrame) -> List[str]:
    return [a for a in ("base",) + analyze.ARMS if a in set(summary["arm"])]


def spread_labels(ys: List[float], gap: float, top: float) -> List[float]:
    """Label heights for values ys: at least `gap` apart, centred on the values, below `top`."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed += np.mean(np.array(ys)[order]) - placed.mean()   # centre the stack on the values
    placed -= max(0.0, placed[-1] - top)                      # keep it inside the plot
    out = np.empty_like(placed)
    out[order] = placed
    return out.tolist()


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _no_data(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ---------------------------------------------------------------------------
# Figure 1: layer profiles
# ---------------------------------------------------------------------------

def layer_profiles(summary: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> Path:
    arms = arms_present(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    offsets = dict(zip(arms, np.linspace(-0.12, 0.12, len(arms))))
    base_rule = decisions[decisions.rule == "a"]
    for ax, task, title, ylabel in (
        (axes[0], "pos", "POS (syntax): accuracy by layer", "UPOS accuracy (%)"),
        (axes[1], "sentiment", "Sentiment (task signal): macro-F1 by layer", "Macro-F1 (%)"),
    ):
        rows = summary[(summary.task == task) & (summary.coords == "full")]
        ends = []
        for arm in arms:
            r = rows[rows.arm == arm].sort_values("layer")
            x = r.layer.to_numpy() + offsets[arm]
            ax.errorbar(x, r["mean"], yerr=r["sd"].fillna(0), color=COLOR[arm], marker=MARKER[arm],
                        markeredgecolor=SURFACE, markeredgewidth=1, elinewidth=1, capsize=0,
                        label=LABEL[arm] + ("" if arm == "base" else f" (n={int(r.n_seeds.max())})"), zorder=3)
            ends.append((float(r["mean"].iloc[-1]), arm))
        if task == "sentiment":
            # Layer 0 [CLS] is constant (chance level, about 19): scale the panel to layers 1-6,
            # where the models differ and the peak is decided, and state layer 0 in a note.
            upper = rows[rows.layer >= 1]
            low = float((upper["mean"] - upper["sd"].fillna(0)).min())
            high = float((upper["mean"] + upper["sd"].fillna(0)).max())
            pad = (high - low) * 0.12
            ax.set_ylim(low - pad, high + pad)
            layer0 = rows[rows.layer == 0]["mean"]
            ax.text(0.98, 0.03, f"Layer 0, off scale: {layer0.min():.1f}–{layer0.max():.1f} (chance level)",
                    transform=ax.transAxes, fontsize=7.5, color=MUTED, va="bottom", ha="right")
        # Headroom above the lines for the peak note.
        lo, hi = ax.get_ylim()
        hi += (hi - lo) * 0.1
        ax.set_ylim(lo, hi)
        # Direct labels at the right end, spread apart so they don't collide.
        heights = spread_labels([y for y, _ in ends], gap=(hi - lo) * 0.065, top=hi)
        for (y, arm), y_text in zip(ends, heights):
            ax.annotate(LABEL[arm], xy=(6 + offsets[arm], y), xytext=(6.35, y_text), color=INK_2, fontsize=8,
                        va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=3))
        # Rule (a): mark base's peak layer.
        if not base_rule.empty:
            peak = int(base_rule[f"{'pos' if task == 'pos' else 'sentiment'}_peak_layer"].iloc[0])
            peak_y = float(rows[(rows.arm == "base") & (rows.layer == peak)]["mean"].iloc[0])
            ax.annotate(f"base peak: layer {peak}", xy=(peak + offsets["base"], peak_y), xytext=(-28, 16),
                        textcoords="offset points", ha="right", fontsize=8, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.set_title(title)
        ax.set_xlabel("Layer (0 = embeddings)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(7))
        ax.set_xlim(-0.4, 6.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(arms), bbox_to_anchor=(0.5, -0.06))
    fig.text(0.0, -0.1, "Mean over training seeds; whiskers ±1 SD (base: one model). All 768 coordinates.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(w_pad=4)
    return save(fig, path)


# ---------------------------------------------------------------------------
# Figure 2: seed-noise rule (b) and PIC-effect check
# ---------------------------------------------------------------------------

def seed_handles(seeds) -> tuple:
    """Legend entries for seeds: marker shape only, in neutral ink, since colour belongs to the model."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=INK_2, marker=SEED_MARKER.get(s, "o"), linestyle="none", markersize=5.5)
               for s in seeds]
    handles.append(Line2D([], [], color=INK, lw=1))
    return handles, [f"seed {s}" for s in seeds] + ["mean over seeds"]


def _difference_panel(ax, bootstrap: pd.DataFrame, a_arm: str, b_arm: str, task: str, coords_by_layer: Dict[int, str],
                      threshold: Dict[int, float], verdict: Dict[int, str], ylabel: str, title: str) -> bool:
    """Draw per-seed differences with CIs; return False (and a "no data" note) if there are none."""
    rows = bootstrap[(bootstrap.task == task) & bootstrap.a.str.match(f"{a_arm}_s\\d+$")
                     & bootstrap.b.str.match(f"{b_arm}_s\\d+$")]
    rows = rows[[coords_by_layer.get(l) == c for l, c in zip(rows.layer, rows.coords)]]
    layers = sorted(coords_by_layer)
    if rows.empty:
        _no_data(ax, "no trained-model results yet")
        ax.set_title(title, pad=22)  # same spacing as panels with verdicts above them
        return False
    seeds = [s for s in analyze.SEEDS if s in set(rows.seed)]
    spread = dict(zip(seeds, np.linspace(-0.18, 0.18, len(seeds)) if len(seeds) > 1 else [0.0]))
    for i, layer in enumerate(layers):
        t = threshold.get(layer, np.nan)
        if not np.isnan(t):
            ax.fill_between([i - 0.4, i + 0.4], -t, t, color=BAND, zorder=0, lw=0)
        for seed in seeds:
            r = rows[(rows.layer == layer) & (rows.seed == seed)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(i + spread[seed], r["diff"], yerr=[[r["diff"] - r["ci_low"]], [r["ci_high"] - r["diff"]]],
                        color=COLOR[a_arm], marker=SEED_MARKER.get(seed, "o"), markeredgecolor=SURFACE,
                        markeredgewidth=1, elinewidth=1.2, capsize=0, zorder=3)
        mean = rows[rows.layer == layer]["diff"].mean()
        ax.hlines(mean, i - 0.3, i + 0.3, color=INK, lw=1, zorder=4)
        ax.text(i, 1.02, verdict.get(layer, ""), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK_2)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=22)
    return True


def seed_noise(bootstrap: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    full = {l: "full" for l in analyze.RULE_B_LAYERS}

    b = decisions[decisions.rule == "b"]
    _difference_panel(axes[0], bootstrap, "mipic", "mrl", "pos", full,
                      dict(zip(b.layer.astype(int), b.threshold)), dict(zip(b.layer.astype(int), b.verdict)),
                      "MIPIC − MRL, accuracy points", "(b) Does MIPIC lose POS information?")
    pic = decisions[decisions.rule == "pic effect"]
    for ax, task, ylabel, title in ((axes[1], "pos", "MRL+PIC − MRL, accuracy points", "PIC effect on POS"),
                                    (axes[2], "sentiment", "MRL+PIC − MRL, macro-F1 points", "PIC effect on sentiment")):
        p = pic[pic.task == task]
        _difference_panel(ax, bootstrap, "mrl_pic", "mrl", task, full,
                          dict(zip(p.layer.astype(int), p.threshold)), dict(zip(p.layer.astype(int), p.verdict)),
                          ylabel, title)
    seeds = [s for s in analyze.SEEDS if s in set(bootstrap.seed)]
    if seeds:
        handles, labels = seed_handles(seeds)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
        labels.append("±T noise band, T = max(2σ of MRL, 0.5)")
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.07))
    fig.text(0.0, -0.11, "Paired within each training seed; whiskers are 95% bootstrap CIs (POS resamples sentences, "
             "sentiment resamples tweets). All 768 coordinates.", fontsize=7.5, color=MUTED)
    fig.tight_layout(w_pad=3)
    return save(fig, path)


# ---------------------------------------------------------------------------
# Figure 3: rule (c) at the PIC checkpoints
# ---------------------------------------------------------------------------

def checkpoints(summary: pd.DataFrame, bootstrap: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    arms = [a for a in ("base", "mrl", "mipic") if a in set(summary.arm)]
    names = [f"({l},{d})" for l, d in probe.PIC_CHECKPOINTS]
    offsets = dict(zip(arms, np.linspace(-0.2, 0.2, len(arms))))
    for arm in arms:
        for coords, filled in (("prefix", False), ("suffix", True)):
            r = summary[(summary.arm == arm) & (summary.task == "pos") & (summary.coords == coords)].sort_values("layer")
            x = np.arange(len(r)) + offsets[arm]
            ax.errorbar(x, r["mean"], yerr=r["sd"].fillna(0), linestyle="none", marker=MARKER[arm], markersize=6.5,
                        color=COLOR[arm], markerfacecolor=COLOR[arm] if filled else SURFACE,
                        markeredgewidth=1.5, elinewidth=1, capsize=0, zorder=3,
                        label=f"{LABEL[arm]}, {coords}")
        pre = summary[(summary.arm == arm) & (summary.task == "pos") & (summary.coords == "prefix")].sort_values("layer")
        suf = summary[(summary.arm == arm) & (summary.task == "pos") & (summary.coords == "suffix")].sort_values("layer")
        ax.vlines(np.arange(len(pre)) + offsets[arm], pre["mean"], suf["mean"], color=COLOR[arm], lw=1, alpha=0.5, zorder=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\nprefix [0:{d}] vs suffix [{d}:768]" for n, (_, d) in zip(names, probe.PIC_CHECKPOINTS)],
                       fontsize=8)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("UPOS accuracy (%)")
    ax.set_title("(c) Where is POS information? Hollow = prefix, filled = suffix", pad=22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(arms), fontsize=7.5, columnspacing=1.2)

    c = decisions[decisions.rule == "c"]
    coords = {l: "prefix" for l, _ in probe.PIC_CHECKPOINTS}
    verdicts = {int(r.layer): r.verdict for r in c.itertuples()}
    drawn = _difference_panel(axes[1], bootstrap, "mipic", "mrl", "sentiment", coords, {}, verdicts,
                              "MIPIC − MRL, macro-F1 points", "(c) Sentiment gain in the prefix")
    if drawn:
        axes[1].set_xticklabels([f"{n}\nprefix [0:{d}]" for n, (_, d) in zip(names, probe.PIC_CHECKPOINTS)], fontsize=8)
    seeds = [s for s in analyze.SEEDS if s in set(bootstrap.seed)]
    if seeds:
        handles, labels = seed_handles(seeds)
        axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(labels), fontsize=7.5)
    fig.text(0.0, -0.1, "Left: mean over training seeds, whiskers ±1 SD. Right: paired within each training seed, "
             "whiskers 95% bootstrap CIs; (c) needs every CI above zero.", fontsize=7.5, color=MUTED)
    fig.tight_layout(w_pad=3)
    return save(fig, path)


def main(argv: Optional[List[str]] = None) -> List[Path]:
    parser = argparse.ArgumentParser(description="Draw the H2 figures from analyze.py's tables.")
    parser.add_argument("--results-dir", type=Path, default=analyze.DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args(argv)

    tables = {}
    for name in ("summary_cells", "bootstrap", "decisions"):
        path = args.results_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run python H2/analyze.py first")
        try:
            tables[name] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            tables[name] = pd.DataFrame()
    bootstrap = tables["bootstrap"]
    if bootstrap.empty:
        bootstrap = pd.DataFrame(columns=["a", "b", "seed", "task", "layer", "coords", "diff", "ci_low", "ci_high"])
        bootstrap = bootstrap.astype({"a": str, "b": str})

    style()
    written = [
        layer_profiles(tables["summary_cells"], tables["decisions"], args.figures_dir / "layer_profiles.png"),
        seed_noise(bootstrap, tables["decisions"], args.figures_dir / "seed_noise.png"),
        checkpoints(tables["summary_cells"], bootstrap, tables["decisions"], args.figures_dir / "checkpoints.png"),
    ]
    for path in written:
        print(f"Saved {path}")
    return written


if __name__ == "__main__":
    main()
