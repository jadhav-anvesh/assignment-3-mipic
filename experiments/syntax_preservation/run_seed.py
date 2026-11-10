"""Run the syntax-preservation pipeline (extract -> probe -> delete features) for every model of one or more seeds.

Models are discovered as <checkpoints-root>/seed<N>/checkpoints/<model>/best.pt and
processed in the order MRL, MRL+PIC, MIPIC, MRL+SIA (the first three are the ones
the decision rules need). Each gets the name <model>_s<N>. For each model:

    1. extract   python experiments/syntax_preservation/extract_features.py --name <name> --checkpoint best.pt
    2. probe     python experiments/syntax_preservation/probe.py --name <name>
    3. clean     delete experiments/syntax_preservation/features/<name>/ (about 6 GB), unless --keep-features

A model counts as done when results/<name>_probe.csv holds all 52 probes; done
models are skipped, so rerunning the same command is safe. Both steps resume on
their own, so an interrupted model continues where it stopped. A failed model is
reported and the run moves on to the next one.

Each model's full output goes to experiments/syntax_preservation/logs/<name>.log (not tracked); the console
shows one line per step. --base runs the untrained backbone as "base" first.

Example, from the repository root:
    python experiments/syntax_preservation/run_seed.py --seeds 42 --dry-run          # show the plan
    python experiments/syntax_preservation/run_seed.py --seeds 42 1 2                # everything, unattended
    python experiments/syntax_preservation/run_seed.py --seeds 42 --models mipic     # one model
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from experiments.syntax_preservation import probe  # noqa: E402

MODEL_ORDER = ("mrl", "mrl_pic", "mipic", "mrl_sia")
DEFAULT_CHECKPOINTS_ROOT = REPO_ROOT / "MRL_LEXIS_H11_checkpoints" / "h11_probe" / "runs"
DEFAULT_LOG_DIR = REPO_ROOT / "experiments" / "syntax_preservation" / "logs"
EXPECTED_PROBES = len(probe.probe_cells()) * (1 + len(probe.PROBE_SEEDS))  # 13 POS + 39 sentiment


@dataclass
class Job:
    name: str
    checkpoint: Optional[Path]  # None = base TinyBERT


def discover(checkpoints_root: Path, seeds: Sequence[int], models: Optional[Sequence[str]], base: bool) -> List[Job]:
    """Jobs in run order; raises if a requested model has no checkpoint."""
    jobs = [Job("base", None)] if base else []
    wanted = list(models) if models else list(MODEL_ORDER)
    unknown = sorted(set(wanted) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {list(MODEL_ORDER)}")
    for seed in seeds:
        for model in MODEL_ORDER:
            if model not in wanted:
                continue
            checkpoint = checkpoints_root / f"seed{seed}" / "checkpoints" / model / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing checkpoint for {model} seed {seed}: {checkpoint}")
            jobs.append(Job(f"{model}_s{seed}", checkpoint))
    return jobs


def probes_done(results_dir: Path, name: str) -> bool:
    path = results_dir / f"{name}_probe.csv"
    return path.exists() and len(pd.read_csv(path)) == EXPECTED_PROBES and (results_dir / f"{name}_preds.npz").exists()


def step_commands(job: Job, args) -> List[tuple]:
    python = [sys.executable, "-u"]
    extract = python + [str(REPO_ROOT / "experiments" / "syntax_preservation" / "extract_features.py"), "--name", job.name,
                        "--out-dir", str(args.features_dir)]
    if job.checkpoint is not None:
        extract += ["--checkpoint", str(job.checkpoint)]
    if args.limit is not None:
        extract += ["--limit", str(args.limit)]
    run_probe = python + [str(REPO_ROOT / "experiments" / "syntax_preservation" / "probe.py"), "--name", job.name,
                          "--features-dir", str(args.features_dir), "--out-dir", str(args.results_dir)]
    if args.threads:
        extract += ["--threads", str(args.threads)]
        run_probe += ["--threads", str(args.threads)]
    return [("extract", extract), ("probe", run_probe)]


def subprocess_runner(command: List[str], log_path: Path) -> int:
    """Run one step with stdout and stderr appended to the model's log; return the exit code."""
    env = dict(os.environ, PYTHONUTF8="1")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n# {datetime.now():%H:%M:%S} $ {' '.join(command)}\n")
        log.flush()
        return subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def run_job(job: Job, args, runner: Callable[[List[str], Path], int] = subprocess_runner) -> str:
    """Run the steps of one model; return its status for the summary."""
    feature_dir = args.features_dir / job.name
    if probes_done(args.results_dir, job.name):
        status = "done earlier"
    else:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.log_dir / f"{job.name}.log"
        for step, command in step_commands(job, args):
            if args.dry_run:
                print(f"  [{job.name}] would run: {' '.join(command)}")
                continue
            start = time.perf_counter()
            print(f"  [{job.name}] {step}: started {datetime.now():%H:%M:%S}", flush=True)
            code = runner(command, log_path)
            minutes = (time.perf_counter() - start) / 60
            if code != 0:
                print(f"  [{job.name}] {step}: FAILED with exit code {code} after {minutes:.1f} min; see {log_path}",
                      flush=True)
                return f"FAILED at {step} (exit {code})"
            print(f"  [{job.name}] {step}: ok, {minutes:.1f} min", flush=True)
        if args.dry_run:
            return "would run"
        if not probes_done(args.results_dir, job.name):
            print(f"  [{job.name}] probe exited 0 but results are incomplete; see {log_path}", flush=True)
            return "FAILED (results incomplete)"
        status = "ok"

    if feature_dir.exists() and not args.keep_features:
        if args.dry_run:
            print(f"  [{job.name}] would delete {feature_dir}")
        else:
            shutil.rmtree(feature_dir)
            print(f"  [{job.name}] deleted features {feature_dir}", flush=True)
    return status


def main(argv: Optional[List[str]] = None, runner: Callable[[List[str], Path], int] = subprocess_runner) -> int:
    parser = argparse.ArgumentParser(description="Run extract -> probe -> clean for every model of the given seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--models", nargs="+", default=None, help=f"Subset of {list(MODEL_ORDER)} (default: all).")
    parser.add_argument("--base", action="store_true", help="Also run the untrained backbone, as 'base', first.")
    parser.add_argument("--checkpoints-root", type=Path, default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--features-dir", type=Path, default=probe.DEFAULT_FEATURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=probe.DEFAULT_OUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--keep-features", action="store_true", help="Keep experiments/syntax_preservation/features/<name>/ after probing.")
    parser.add_argument("--threads", type=int, default=None, help="torch threads for both steps.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: extract only the first N sentences of each split. "
                             "Use with other --features-dir/--results-dir.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.seeds and not args.base:
        parser.error("give --seeds and/or --base")
    if args.limit is not None and args.results_dir == probe.DEFAULT_OUT_DIR:
        parser.error("--limit writes smoke-test results; give a separate --results-dir")

    jobs = discover(args.checkpoints_root, args.seeds, args.models, args.base)
    print(f"{len(jobs)} models: {', '.join(j.name for j in jobs)}" + ("  (dry run)" if args.dry_run else ""),
          flush=True)
    summary = []
    for number, job in enumerate(jobs, start=1):
        print(f"[{number}/{len(jobs)}] {job.name}" + (f" from {job.checkpoint}" if job.checkpoint else " (base TinyBERT)"),
              flush=True)
        start = time.perf_counter()
        status = run_job(job, args, runner)
        summary.append((job.name, status, (time.perf_counter() - start) / 60))

    print("\nSummary:")
    print(f"{'model':<14} {'minutes':>8}  status")
    for name, status, minutes in summary:
        print(f"{name:<14} {minutes:8.1f}  {status}")
    failed = [name for name, status, _ in summary if status.startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}. Rerun the same command to retry; finished work is kept.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
