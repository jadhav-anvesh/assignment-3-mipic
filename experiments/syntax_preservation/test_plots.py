"""Tests for the syntax-preservation figures: they render from analyze.py's tables, with all models or base alone.

Run from the repository root:  python -m pytest experiments/syntax_preservation/test_plots.py -q
No model needed; uses the synthetic results of test_analyze.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.syntax_preservation import analyze, plots  # noqa: E402
import test_analyze as synthetic  # noqa: E402

FIGURES = ("layer_profiles.png", "seed_noise.png", "checkpoints.png")


def analyze_into(folder: Path, monkeypatch) -> None:
    monkeypatch.setattr(analyze, "pos_sentence_index", lambda n: synthetic.POS_OWNER)
    analyze.main(["--results-dir", str(folder), "--report", str(folder / "RESULTS.md"), "--n-resamples", "100"])


def test_spread_labels_keeps_order_gap_and_ceiling():
    heights = plots.spread_labels([5.0, 5.0, 5.1, 9.0], gap=1.0, top=10.0)
    assert heights[0] < heights[2] and heights[1] < heights[2]  # ties keep data order, order kept
    ordered = sorted(heights)
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(ordered, ordered[1:]))
    assert max(heights) <= 10.0
    assert plots.spread_labels([3.0, 7.0], gap=1.0, top=10.0) == [3.0, 7.0]  # already apart: unchanged


def test_all_figures_render_with_every_model(tmp_path, monkeypatch):
    synthetic.build(tmp_path, mipic_pos_drop=0.0, mipic_sentiment_prefix=0.75, pic_sentiment_l1=0.8)
    for i, seed in enumerate((42, 1, 2)):
        synthetic.write_model(tmp_path, f"mrl_sia_s{seed}", {}, seed=40 + i)
    analyze_into(tmp_path, monkeypatch)
    written = plots.main(["--results-dir", str(tmp_path), "--figures-dir", str(tmp_path / "figures")])
    assert [p.name for p in written] == list(FIGURES)
    assert all(p.stat().st_size > 20_000 for p in written)


def test_figures_render_with_base_only(tmp_path, monkeypatch):
    synthetic.write_model(tmp_path, "base", synthetic.BASE_ACC, seed=7)
    analyze_into(tmp_path, monkeypatch)
    written = plots.main(["--results-dir", str(tmp_path), "--figures-dir", str(tmp_path / "figures")])
    assert all(p.exists() for p in written)


def test_missing_tables_are_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="run python experiments/syntax_preservation/analyze.py first"):
        plots.main(["--results-dir", str(tmp_path), "--figures-dir", str(tmp_path / "figures")])
