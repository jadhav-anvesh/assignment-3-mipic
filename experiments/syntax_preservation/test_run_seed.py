"""Tests for the syntax-preservation launcher, with a fake checkpoint tree and a stand-in for the two steps.

Run from the repository root:  python -m pytest experiments/syntax_preservation/test_run_seed.py -q
No model needed; runs in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.syntax_preservation import run_seed  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "runs"
    for seed in (42, 1):
        for model in run_seed.MODEL_ORDER:
            path = root / f"seed{seed}" / "checkpoints" / model / "best.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"")
    return tmp_path


def args_for(tree: Path, *extra: str) -> List[str]:
    return ["--checkpoints-root", str(tree / "runs"), "--features-dir", str(tree / "features"),
            "--results-dir", str(tree / "results"), "--log-dir", str(tree / "logs"), *extra]


class FakeSteps:
    """Stands in for extract_features.py and probe.py; can be told to fail one step."""

    def __init__(self, fail: tuple = ()):
        self.calls: List[tuple] = []
        self.fail = fail

    def __call__(self, command: List[str], log_path: Path) -> int:
        name = command[command.index("--name") + 1]
        step = "extract" if command[2].endswith("extract_features.py") else "probe"
        self.calls.append((name, step))
        if (name, step) in self.fail:
            return 3
        if step == "extract":
            out = Path(command[command.index("--out-dir") + 1]) / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "ud_train_layer0.npy").write_bytes(b"x")
        else:
            out = Path(command[command.index("--out-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"task": ["pos"] * run_seed.EXPECTED_PROBES}).to_csv(out / f"{name}_probe.csv", index=False)
            np.savez(out / f"{name}_preds.npz", x=np.zeros(1))
        return 0


def test_discover_orders_models_and_checks_checkpoints(tree):
    jobs = run_seed.discover(tree / "runs", [42, 1], None, base=True)
    assert [j.name for j in jobs] == ["base", "mrl_s42", "mrl_pic_s42", "mipic_s42", "mrl_sia_s42",
                                      "mrl_s1", "mrl_pic_s1", "mipic_s1", "mrl_sia_s1"]
    assert jobs[0].checkpoint is None
    assert [j.name for j in run_seed.discover(tree / "runs", [1], ["mipic", "mrl"], False)] == ["mrl_s1", "mipic_s1"]
    with pytest.raises(FileNotFoundError, match="seed 2"):
        run_seed.discover(tree / "runs", [2], None, False)
    with pytest.raises(ValueError, match="unknown models"):
        run_seed.discover(tree / "runs", [42], ["bert"], False)


def test_runs_both_steps_then_deletes_features(tree):
    steps = FakeSteps()
    assert run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl"), runner=steps) == 0
    assert steps.calls == [("mrl_s42", "extract"), ("mrl_s42", "probe")]
    assert not (tree / "features" / "mrl_s42").exists()
    assert (tree / "results" / "mrl_s42_probe.csv").exists()


def test_finished_models_are_skipped(tree):
    steps = FakeSteps()
    run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl"), runner=steps)
    steps.calls.clear()
    assert run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl", "mipic"), runner=steps) == 0
    assert steps.calls == [("mipic_s42", "extract"), ("mipic_s42", "probe")]


def test_incomplete_results_are_not_treated_as_done(tree):
    (tree / "results").mkdir()
    pd.DataFrame({"task": ["pos"] * 5}).to_csv(tree / "results" / "mrl_s42_probe.csv", index=False)
    np.savez(tree / "results" / "mrl_s42_preds.npz", x=np.zeros(1))
    steps = FakeSteps()
    run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl"), runner=steps)
    assert ("mrl_s42", "probe") in steps.calls


def test_a_failure_is_reported_and_the_run_continues(tree, capsys):
    steps = FakeSteps(fail=(("mrl_pic_s42", "extract"),))
    code = run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl", "mrl_pic", "mipic"), runner=steps)
    assert code == 1
    assert ("mrl_pic_s42", "probe") not in steps.calls  # no probe after a failed extraction
    assert ("mipic_s42", "probe") in steps.calls        # the next model still runs
    out = capsys.readouterr().out
    assert "FAILED at extract (exit 3)" in out and "1 failed: mrl_pic_s42" in out


def test_keep_features_and_dry_run(tree, capsys):
    steps = FakeSteps()
    run_seed.main(args_for(tree, "--seeds", "42", "--models", "mrl", "--keep-features"), runner=steps)
    assert (tree / "features" / "mrl_s42").exists()

    steps.calls.clear()
    run_seed.main(args_for(tree, "--seeds", "1", "--dry-run"), runner=steps)
    assert steps.calls == []
    out = capsys.readouterr().out
    assert out.count("would run:") == 8 and "(dry run)" in out


def test_limit_needs_a_separate_results_dir(tree):
    with pytest.raises(SystemExit):
        run_seed.main(["--checkpoints-root", str(tree / "runs"), "--seeds", "42", "--limit", "5"], runner=FakeSteps())
