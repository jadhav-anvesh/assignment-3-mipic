"""Extra paired bootstraps for FINDINGS 6 and 7: KL-on MIPIC vs MRL/base, and the SIA x PIC interaction.

Same method as bootstrap.py: one index matrix of --n-resamples x n_rows from
default_rng(--seed), shared by every (layer, width) cell and by every model,
and the same macro-F1 (bootstrap.macro_f1_rows).

1. Pairwise comparisons, run through bootstrap.main itself, so the output is
   exactly what bootstrap.py writes:
       mipic_kl{suffix} vs mrl{suffix}  -> bootstrap_mipic_kl_s42_vs_mrl_s42.csv
       mipic_kl{suffix} vs base         -> bootstrap_mipic_kl_s42_vs_base.csv

2. The SIA x PIC interaction (H16) at every (layer, width) cell:
       interaction = F1(mipic) - F1(mrl_sia) - F1(mrl_pic) + F1(mrl)
   i.e. MIPIC's gain over MRL minus the SIA gain and the PIC gain. All four
   models are scored on the same resampled rows, and the 95% interval is the
   percentile interval of the resampled interaction.
       -> bootstrap_interaction{suffix}.csv

F1 values are in points (x100).

Example, from the repository root:
    python experiments/layerwise_probe/bootstrap_extra.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.layerwise_probe import bootstrap  # noqa: E402
from experiments.layerwise_probe.probe_grid import DIMS, ROWS, prediction_key  # noqa: E402

# Signs of each arm in the interaction contrast.
INTERACTION = (("mipic", 1), ("mrl_sia", -1), ("mrl_pic", -1), ("mrl", 1))


def interaction_table(results_dir: Path, suffix: str, n_resamples: int, seed: int) -> pd.DataFrame:
    names = {arm: f"{arm}{suffix}" for arm, _ in INTERACTION}
    preds = {arm: bootstrap.load_preds(results_dir, name) for arm, name in names.items()}
    labels = preds["mrl"]["labels"]
    for arm, p in preds.items():
        if not np.array_equal(labels, p["labels"]):
            raise ValueError(f"{names[arm]} was scored on different test labels; a paired bootstrap is invalid")

    # Identical to bootstrap.py: same generator, same call, same shape.
    n = len(labels)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    y_boot = labels[idx]

    keys = [(row, dim) for row in ROWS for dim in DIMS]
    classes = np.unique(np.concatenate(
        [labels] + [preds[arm][prediction_key(r, d)] for arm in preds for r, d in keys]
    ))

    out = []
    for row, dim in keys:
        key = prediction_key(row, dim)
        observed, resampled = {}, {}
        for arm, _ in INTERACTION:
            p = preds[arm][key]
            observed[arm] = float(bootstrap.macro_f1_rows(labels, p, classes)) * 100
            resampled[arm] = bootstrap.macro_f1_rows(y_boot, p[idx], classes) * 100
        value = sum(sign * observed[arm] for arm, sign in INTERACTION)
        boot = sum(sign * resampled[arm] for arm, sign in INTERACTION)
        low, high = np.percentile(boot, [2.5, 97.5])
        out.append({
            "layer": str(row),
            "dim": dim,
            "f1_mrl": observed["mrl"],
            "f1_mrl_sia": observed["mrl_sia"],
            "f1_mrl_pic": observed["mrl_pic"],
            "f1_mipic": observed["mipic"],
            "gain_sia": observed["mrl_sia"] - observed["mrl"],
            "gain_pic": observed["mrl_pic"] - observed["mrl"],
            "gain_mipic": observed["mipic"] - observed["mrl"],
            "interaction": value,
            "ci_low": low,
            "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "boot_se": float(np.std(boot, ddof=1)),
        })

    table = pd.DataFrame(out)
    table.insert(0, "models", " ".join(f"{'+' if s > 0 else '-'}{names[a]}" for a, s in INTERACTION))
    table["n_resamples"] = n_resamples
    table["seed"] = seed
    return table


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="KL-on pairwise bootstraps and the SIA x PIC interaction bootstrap.")
    parser.add_argument("--suffix", default="_s42", help="Seed suffix of the trained models' preds names.")
    parser.add_argument("--base", default="base", help="Preds name of the untrained backbone.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "experiments" / "layerwise_probe" / "results")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Only the interaction; for seeds without a KL-on MIPIC run.")
    args = parser.parse_args(argv)

    common = ["--results-dir", str(args.results_dir), "--n-resamples", str(args.n_resamples), "--seed", str(args.seed)]
    if not args.skip_pairwise:
        for b in (f"mrl{args.suffix}", args.base):
            bootstrap.main(["--a", f"mipic_kl{args.suffix}", "--b", b] + common)
            print()

    table = interaction_table(args.results_dir, args.suffix, args.n_resamples, args.seed)
    out_path = args.results_dir / f"bootstrap_interaction{args.suffix}.csv"
    table.to_csv(out_path, index=False)

    show = table.copy()
    show["cell"] = show.apply(
        lambda r: f"{r['interaction']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]{'*' if r['excludes_zero'] else ''}", axis=1
    )
    pivot = show.pivot(index="layer", columns="dim", values="cell").reindex([str(r) for r in ROWS])
    print(f"SIA x PIC interaction ({table['models'].iloc[0]}), macro-F1 points, 95% CI from "
          f"{args.n_resamples} paired resamples (seed {args.seed}); * = CI excludes 0")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(pivot.to_string())
    print(f"\n{int(table['excludes_zero'].sum())} of {len(table)} cells exclude zero")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
