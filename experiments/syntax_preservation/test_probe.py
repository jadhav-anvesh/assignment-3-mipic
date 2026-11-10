"""Tests for the syntax-preservation probes, on synthetic feature folders with known signal.

Run from the repository root:  python -m pytest experiments/syntax_preservation/test_probe.py -q
No model or download needed; runs in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.syntax_preservation import probe  # noqa: E402

SIZES = {"ud_train": 600, "ud_dev": 200, "ud_test": 200, "tweet_train": 600, "tweet_test": 200}
PREFIX_SIGNAL = 3     # coordinate inside every checkpoint prefix [0:16]
SUFFIX_SIGNAL = 700   # coordinate inside every suffix [d:768]


NOISE = slice(100, 120)  # noisy coordinates, inside every suffix


def make_features(folder: Path, seed: int = 0) -> None:
    """Labels are readable from coordinate 3 at layer 1 and from coordinate 700 at layer 2.

    Coordinates 100-119 are noise and the rest are constant. (With hundreds of
    noise coordinates and only a few hundred rows, a linear probe overfits the
    noise and misses even a strong planted signal; the real data has 200k rows.)
    Layer 0 tweet [CLS] is constant, as in the real model.
    """
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    for part, n in SIZES.items():
        n_classes = 5 if part.startswith("ud") else 3
        labels = rng.integers(0, n_classes, size=n)
        for layer in probe.PROFILE_LAYERS:
            x = np.zeros((n, probe.HIDDEN), dtype=np.float32)
            x[:, NOISE] = rng.normal(size=(n, NOISE.stop - NOISE.start))
            if layer == 1:
                x[:, PREFIX_SIGNAL] = labels * 6.0 + rng.normal(size=n)
            if layer == 2:
                x[:, SUFFIX_SIGNAL] = labels * 6.0 + rng.normal(size=n)
            if layer == 0 and part.startswith("tweet"):
                x[:] = x[0]
            np.save(folder / f"{part}_layer{layer}.npy", x)
        name = f"{part}_upos" if part.startswith("ud") else f"{part}_labels"
        np.save(folder / f"{name}.npy", labels.astype(np.int8 if part.startswith("ud") else np.int64))
    meta = {"completed": {part: {"rows": n} for part, n in SIZES.items()}}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_cells():
    cells = probe.probe_cells()
    assert len(cells) == 13
    assert [c.layer for c in cells if c.coords == "full"] == list(range(7))
    assert probe.Cell(2, "prefix", 0, 32) in cells and probe.Cell(2, "suffix", 32, 768) in cells
    assert all(c.stop - c.start == 768 - 16 for c in cells if c.layer == 1 and c.coords == "suffix")


def test_validation_split_is_stratified_seeded_and_disjoint():
    labels = np.repeat([0, 1, 2], [500, 300, 200])
    fit_a, val_a = probe.validation_split(labels, 0)
    fit_b, val_b = probe.validation_split(labels, 0)
    _, val_c = probe.validation_split(labels, 1)
    assert np.array_equal(val_a, val_b) and not np.array_equal(val_a, val_c)
    assert len(np.intersect1d(fit_a, val_a)) == 0 and len(fit_a) + len(val_a) == len(labels)
    assert np.bincount(labels[val_a]).tolist() == [50, 30, 20]


def test_standardize_uses_fitting_split_only():
    x_fit = np.array([[0.0, 5.0], [2.0, 5.0]])
    x_other = np.array([[4.0, 7.0]])
    fit_scaled, other_scaled = probe.standardize(x_fit, x_other)
    assert np.allclose(fit_scaled, [[-1.0, 0.0], [1.0, 0.0]])
    assert np.allclose(other_scaled, [[3.0, 2.0]])  # constant column centred, not scaled


@pytest.mark.parametrize("C", [0.01, 1.0])
def test_linear_probe_matches_sklearn_logistic_regression(C):
    """Same objective as evaluate.py's LogisticRegression, so the same model up to solver tolerance."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 3000)
    x = rng.normal(size=(3000, 40)) + np.eye(5)[y] @ rng.normal(size=(5, 40))
    (x_std,) = probe.standardize(x)
    ours = probe.LinearProbe(C=C, tol=1e-6).fit(x_std, y)
    ref = LogisticRegression(C=C, max_iter=1000, tol=1e-8).fit(x_std.astype(np.float64), y)
    assert np.mean(ours.predict(x_std) == ref.predict(x_std)) > 0.995
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=2e-3 * np.abs(ref.coef_).max())
    # Same predicted probabilities (intercepts alone are only defined up to a common shift).
    scores = ours.decision_function(x_std).astype(np.float64)
    proba = np.exp(scores - scores.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(proba, ref.predict_proba(x_std.astype(np.float64)), atol=2e-3)
    assert ours.n_iter_ < probe.PROBE_MAX_ITER


def test_linear_probe_keeps_label_values():
    y = np.array([3, 7, 3, 7, 9, 9], dtype=np.int8)
    x = np.eye(3)[[0, 1, 0, 1, 2, 2]].astype(np.float32)
    model = probe.LinearProbe(C=10.0).fit(x, y)
    assert model.classes_.tolist() == [3, 7, 9]
    assert model.predict(x).tolist() == y.tolist()


def test_fit_and_select_prefers_best_validation_c():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=(400, 20)) + y[:, None] * 0.3
    result = probe.fit_and_select(x[:200], y[:200], x[200:300], y[200:300], x[300:], probe.accuracy)
    assert [g["C"] for g in result["grid"]] == sorted(probe.C_GRID)
    best = max(g["val_score"] for g in result["grid"])
    assert result["val_score"] == best
    tied = [g["C"] for g in result["grid"] if g["val_score"] == best]
    assert result["C"] == min(tied)
    assert len(result["test_pred"]) == 100


@pytest.fixture(scope="module")
def probed(tmp_path_factory):
    root = tmp_path_factory.mktemp("h2")
    make_features(root / "features" / "toy")
    args = ["--name", "toy", "--features-dir", str(root / "features"), "--out-dir", str(root / "results"),
            "--threads", "1"]
    probe.main(args)
    return root, args


def test_outputs_and_planted_signal(probed):
    root, _ = probed
    table = pd.read_csv(root / "results" / "toy_probe.csv")
    assert len(table) == 13 + 13 * 3  # POS once per cell, sentiment once per probe seed
    with np.load(root / "results" / "toy_preds.npz") as stored:  # closed again, so a rerun can replace it
        preds = dict(stored)
    assert set(preds) == {probe.prediction_key(r.task, probe.Cell(r.layer, r.coords, r.start, r.stop), r.probe_seed)
                                for r in table.itertuples()} | {"pos_labels", "sentiment_labels"}
    assert len(pd.read_csv(root / "results" / "toy_cgrid.csv")) == len(table) * len(probe.C_GRID)

    def score(task, layer, coords):
        rows = table[(table.task == task) & (table.layer == layer) & (table.coords == coords)]
        return rows["test_accuracy"].mean()

    for task in ("pos", "sentiment"):
        assert score(task, 1, "prefix") > 0.95 and score(task, 1, "suffix") < 0.5   # signal in the prefix only
        assert score(task, 2, "suffix") > 0.95 and score(task, 2, "prefix") < 0.5   # signal in the suffix only
        assert score(task, 1, "full") > 0.95 and score(task, 2, "full") > 0.95
    # Constant layer 0 [CLS]: the probe can only predict one class.
    for seed in probe.PROBE_SEEDS:
        assert len(np.unique(preds[f"sentiment_L0_full_s{seed}"])) == 1


def test_rerun_resumes_without_refitting(probed):
    root, args = probed
    progress = root / "features" / "toy" / "probe_progress"
    stamps = {p.name: p.stat().st_mtime_ns for p in progress.glob("*.npz")}
    assert len(stamps) == 13 + 13 * 3
    probe.main(args)
    assert {p.name: p.stat().st_mtime_ns for p in progress.glob("*.npz")} == stamps


def test_changed_settings_are_refused(probed):
    _, args = probed
    with pytest.raises(ValueError, match="other settings"):
        probe.main(args + ["--c-grid", "1.0"])


def test_missing_parts_are_reported(tmp_path):
    folder = tmp_path / "features" / "half"
    make_features(folder)
    (folder / "meta.json").write_text(json.dumps({"completed": {"ud_train": {}}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not extracted yet"):
        probe.main(["--name", "half", "--features-dir", str(tmp_path / "features"), "--out-dir", str(tmp_path)])
