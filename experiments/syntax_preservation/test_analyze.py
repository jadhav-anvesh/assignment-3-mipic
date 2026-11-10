"""Tests for the syntax-preservation analysis, on synthetic results with known effects.

Run from the repository root:  python -m pytest experiments/syntax_preservation/test_analyze.py -q
No model needed; runs in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.syntax_preservation import analyze, probe  # noqa: E402

N_SENTENCES, WORDS_PER_SENTENCE, N_TWEETS = 300, 8, 1200
POS_LABELS = np.random.default_rng(100).integers(0, 17, N_SENTENCES * WORDS_PER_SENTENCE).astype(np.int8)
POS_OWNER = np.repeat(np.arange(N_SENTENCES), WORDS_PER_SENTENCE)
SENT_LABELS = np.random.default_rng(101).integers(0, 3, N_TWEETS).astype(np.int64)


def noisy(labels: np.ndarray, accuracy: float, rng, n_classes: int) -> np.ndarray:
    """Predictions with exactly round(accuracy * n) correct; wrong ones get another class."""
    pred = labels.copy()
    wrong = rng.choice(len(labels), size=int(round((1 - accuracy) * len(labels))), replace=False)
    pred[wrong] = (labels[wrong] + rng.integers(1, n_classes, len(wrong))) % n_classes
    return pred


def write_model(folder: Path, name: str, accuracy: Dict[tuple, float], seed: int,
                same_as: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """One model's probe.csv and preds.npz. accuracy[(task, layer, coords)] overrides the default 0.6."""
    rng = np.random.default_rng(seed)
    rows, preds = [], {}
    for task in probe.TASKS:
        labels, n_classes = (POS_LABELS, 17) if task == "pos" else (SENT_LABELS, 3)
        for cell in probe.probe_cells():
            for probe_seed in ((0,) if task == "pos" else probe.PROBE_SEEDS):
                key = probe.prediction_key(task, cell, probe_seed)
                if same_as is not None and (task, cell.layer, cell.coords) not in accuracy:
                    pred = same_as[key]
                else:
                    pred = noisy(labels, accuracy.get((task, cell.layer, cell.coords), 0.6), rng, n_classes)
                preds[key] = pred.astype(np.int8)
                rows.append({"task": task, "layer": cell.layer, "coords": cell.coords, "start": cell.start,
                             "stop": cell.stop, "probe_seed": probe_seed,
                             "test_accuracy": float(np.mean(pred == labels)),
                             "test_macro_f1": float(analyze.macro_f1_rows(labels, pred, np.unique(labels)))})
    pd.DataFrame(rows).to_csv(folder / f"{name}_probe.csv", index=False)
    np.savez(folder / f"{name}_preds.npz", pos_labels=POS_LABELS, sentiment_labels=SENT_LABELS, **preds)
    return preds


def run(folder: Path):
    models = analyze.load_models(folder)
    boot = analyze.Bootstrap(POS_LABELS, POS_OWNER, SENT_LABELS, n_resamples=300)
    out = analyze.analyze(models, boot)
    return models, out, out["decisions"]


def verdict(decisions: pd.DataFrame, rule: str) -> str:
    return decisions[decisions.rule == rule].verdict.iloc[0]


# --------------------------------------------------------------------------- unit tests

def test_macro_f1_rows_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    assert np.isclose(analyze.macro_f1_rows(y, p, np.arange(3)), f1_score(y, p, average="macro"))


def test_seed_noise_rule():
    rule = analyze.seed_noise
    # sigma = 0.1 -> T = max(0.2, 0.5) = 0.5 (the floor)
    assert rule([-0.3, -0.4, -0.2], [(-0.5, -0.1)] * 3, [90.0, 90.1, 90.2])["verdict"] == "within noise"
    assert rule([-2.0, -1.5, -1.8], [(-2.5, -1.0)] * 3, [90.0, 90.1, 90.2])["verdict"] == "damage"
    # Large drop, but one seed's CI crosses zero -> not damage
    assert rule([-2.0, -1.5, -1.8], [(-2.5, -1.0), (-2.0, 0.1), (-2.5, -1.0)], [90.0, 90.1, 90.2])["verdict"] == "inconclusive"
    # sigma = 1 -> T = 2: a 1.5-point drop is within noise even with every CI below zero
    result = rule([-1.5, -1.5, -1.5], [(-2.0, -1.0)] * 3, [89.0, 90.0, 91.0])
    assert np.isclose(result["threshold"], 2.0) and result["verdict"] == "within noise"
    up = rule([2.0, 2.5, 3.0], [(1.0, 4.0)] * 3, [90.0, 90.1, 90.2])
    assert up["measurable_effect"] and up["verdict"] == "inconclusive"
    assert rule([-1.0], [(-2.0, -0.5)], [90.0])["verdict"] == "pending"


def test_parse_name():
    assert analyze.parse_name("mrl_pic_s42") == ("mrl_pic", 42)
    assert analyze.parse_name("base") == ("base", None)
    with pytest.raises(ValueError):
        analyze.parse_name("mipic_kl_s42")


def test_pos_bootstrap_resamples_sentences():
    boot = analyze.Bootstrap(POS_LABELS, POS_OWNER, SENT_LABELS, n_resamples=200)
    pred = noisy(POS_LABELS, 0.8, np.random.default_rng(1), 17)[None, :]
    observed, resampled = boot.pos(pred)
    assert np.isclose(observed, 80.0)
    assert resampled.shape == (200,) and 75 < resampled.mean() < 85


# --------------------------------------------------------------------------- scenarios

BASE_ACC = {("pos", 3, "full"): 0.9, ("sentiment", 6, "full"): 0.9}
MRL_POS = {1: 0.85, 2: 0.88, 3: 0.90}


def mrl_accuracy(seed_offset: float) -> dict:
    acc = {("pos", l, "full"): a + seed_offset for l, a in MRL_POS.items()}
    acc.update({("sentiment", l, "prefix"): 0.5 for l, _ in probe.PIC_CHECKPOINTS})
    return acc


def build(folder: Path, mipic_pos_drop: float, mipic_sentiment_prefix: float, pic_sentiment_l1: Optional[float],
          seeds=(42, 1, 2)) -> None:
    write_model(folder, "base", BASE_ACC, seed=7)
    for i, seed in enumerate(seeds):
        offset = (0.0, 0.002, -0.002)[i]  # MRL's seed-to-seed variation: sigma = 0.2 points
        mrl = write_model(folder, f"mrl_s{seed}", mrl_accuracy(offset), seed=10 + i)
        mipic_acc = {("sentiment", l, "prefix"): mipic_sentiment_prefix for l, _ in probe.PIC_CHECKPOINTS}
        mipic_acc.update({("pos", l, "suffix"): 0.9 for l, _ in probe.PIC_CHECKPOINTS})
        mipic_acc.update({("pos", l, "prefix"): 0.5 for l, _ in probe.PIC_CHECKPOINTS})
        if mipic_pos_drop:
            mipic_acc.update({("pos", l, "full"): a + offset - mipic_pos_drop for l, a in MRL_POS.items()})
        write_model(folder, f"mipic_s{seed}", mipic_acc, seed=20 + i, same_as=mrl)
        pic_acc = {} if pic_sentiment_l1 is None else {("sentiment", 1, "full"): pic_sentiment_l1}
        write_model(folder, f"mrl_pic_s{seed}", pic_acc, seed=30 + i, same_as=mrl)


def test_supported_scenario(tmp_path):
    build(tmp_path, mipic_pos_drop=0.0, mipic_sentiment_prefix=0.75, pic_sentiment_l1=0.8)
    models, out, d = run(tmp_path)
    assert verdict(d, "a") == "supported"
    assert d[d.rule == "a"].pos_peak_layer.iloc[0] == 3 and d[d.rule == "a"].sentiment_peak_layer.iloc[0] == 6
    assert verdict(d, "b (overall)") == "supported"          # MIPIC's POS predictions equal MRL's
    assert verdict(d, "c (overall)") == "supported"
    assert verdict(d, "pic effect (overall)") == "PIC has a measurable effect"
    # Every trained arm is compared with MRL of its seed and with base.
    pairs = set(zip(out["bootstrap"].a, out["bootstrap"].b))
    assert ("mipic_s1", "mrl_s1") in pairs and ("mrl_s1", "base") in pairs and ("mipic_s1", "mrl_s42") not in pairs
    report = analyze.render_report(models, out)
    assert "**Rule (b): supported.**" in report and "Provisional" not in report


def test_damage_and_no_effect_scenario(tmp_path):
    build(tmp_path, mipic_pos_drop=0.05, mipic_sentiment_prefix=0.5, pic_sentiment_l1=None)
    _, _, d = run(tmp_path)
    assert set(d[d.rule == "b"].verdict) == {"damage"}
    assert verdict(d, "b (overall)") == "falsified (damage)"
    assert verdict(d, "c (overall)") == "supported at 0 of 3 checkpoints"   # no sentiment gain in the prefix
    assert verdict(d, "pic effect (overall)") == "falsified (no measurable effect)"


def test_one_seed_is_pending_and_provisional(tmp_path):
    build(tmp_path, mipic_pos_drop=0.0, mipic_sentiment_prefix=0.75, pic_sentiment_l1=0.8, seeds=(42,))
    models, out, d = run(tmp_path)
    assert verdict(d, "b (overall)") == "pending"
    assert verdict(d, "pic effect (overall)") == "pending"
    assert verdict(d, "c (overall)") == "supported"   # (c) needs no seed SD, but is marked provisional
    assert bool(d[d.rule == "c (overall)"].provisional.iloc[0])
    assert "**Provisional:** 1 of 3" in analyze.render_report(models, out)


def test_base_only(tmp_path):
    write_model(tmp_path, "base", BASE_ACC, seed=7)
    models, out, d = run(tmp_path)
    assert verdict(d, "a") == "supported"
    assert verdict(d, "b (overall)") == "pending" and verdict(d, "c (overall)") == "pending"
    analyze.render_report(models, out)  # renders without trained models


def test_main_writes_outputs(tmp_path, monkeypatch):
    build(tmp_path, mipic_pos_drop=0.0, mipic_sentiment_prefix=0.75, pic_sentiment_l1=0.8, seeds=(42, 1))
    monkeypatch.setattr(analyze, "pos_sentence_index", lambda n: POS_OWNER)
    analyze.main(["--results-dir", str(tmp_path), "--report", str(tmp_path / "RESULTS.md"), "--n-resamples", "200"])
    for name in ("summary_cells.csv", "bootstrap.csv", "decisions.csv", "RESULTS.md"):
        assert (tmp_path / name).exists()
    summary = pd.read_csv(tmp_path / "summary_cells.csv")
    assert set(summary.arm) == {"base", "mrl", "mipic", "mrl_pic"}
    assert summary[summary.arm == "mrl"].n_seeds.eq(2).all()
