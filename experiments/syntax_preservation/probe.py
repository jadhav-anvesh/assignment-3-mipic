"""Fit the syntax-preservation linear probes on one model's features: POS and sentiment, by layer and coordinate block.

Reads experiments/syntax_preservation/features/<name>/ (written by extract_features.py) and fits, per cell:

    Layer profiles      layers 0-6, all 768 coordinates             ("full")
    Prefix vs suffix    at the PIC checkpoints (1,16), (2,32), (3,64):
                        coordinates [0:d] ("prefix") and [d:768] ("suffix")

for two tasks:

    pos        UPOS (17 tags) on UD EWT word vectors. Fit on train, choose C on
               dev, report on test. Metric for choosing C: accuracy.
    sentiment  3-class tweet sentiment on [CLS] vectors. For each probe seed
               (0, 1, 2) a stratified 10% of tweet_train is held out: fit on
               the other 90%, choose C on the 10%, report on tweet_test.
               Metric for choosing C: macro-F1.

The probe: features are standardized with the mean and std of the fitting split
only, then an L2-regularised multinomial logistic regression is fitted for each
C in the grid: the objective of evaluate.py's LogisticRegression (lbfgs, L2,
max_iter=1000), solved with full-batch L-BFGS in PyTorch (LinearProbe below).
The C with the best validation score is chosen (ties go to the smaller C, i.e.
stronger regularization) and that fitted model's test predictions are kept.
Every fit starts from zero.

Fits run one at a time with a fixed number of torch threads (--threads, recorded
in the output).

Outputs, in experiments/syntax_preservation/results/:
    <name>_probe.csv   one row per (task, cell, probe seed): chosen C, validation
                       score, test accuracy and macro-F1, convergence
    <name>_cgrid.csv   validation score of every C, for every row above
    <name>_preds.npz   test predictions of the chosen model per row, keyed
                       "<task>_L<layer>_<coords>_s<seed>", plus "pos_labels" and
                       "sentiment_labels"

Progress is saved per cell in experiments/syntax_preservation/features/<name>/probe_progress/, so an
interrupted run resumes where it stopped.

Example, from the repository root:
    python experiments/syntax_preservation/probe.py --name base
    python experiments/syntax_preservation/probe.py --name base --tasks sentiment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HIDDEN = 768
PROFILE_LAYERS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
PIC_CHECKPOINTS: Tuple[Tuple[int, int], ...] = ((1, 16), (2, 32), (3, 64))
C_GRID: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
PROBE_SEEDS: Tuple[int, ...] = (0, 1, 2)
VAL_FRACTION = 0.1
TASKS = ("pos", "sentiment")
PROBE_MAX_ITER = 1000  # as in evaluate.py / layerwise_probe/probe_grid.py

DEFAULT_FEATURES_DIR = REPO_ROOT / "experiments" / "syntax_preservation" / "features"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "syntax_preservation" / "results"


@dataclass(frozen=True)
class Cell:
    layer: int
    coords: str  # "full", "prefix" or "suffix"
    start: int
    stop: int

    @property
    def key(self) -> str:
        return f"L{self.layer}_{self.coords}"


def probe_cells() -> List[Cell]:
    """The 13 cells probed for every task: 7 layer-profile cells, then prefix and suffix at 3 checkpoints."""
    cells = [Cell(layer, "full", 0, HIDDEN) for layer in PROFILE_LAYERS]
    for layer, dim in PIC_CHECKPOINTS:
        cells.append(Cell(layer, "prefix", 0, dim))
        cells.append(Cell(layer, "suffix", dim, HIDDEN))
    return cells


def prediction_key(task: str, cell: Cell, seed: int) -> str:
    return f"{task}_{cell.key}_s{seed}"


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


SELECTION_METRIC = {"pos": ("accuracy", accuracy), "sentiment": ("macro_f1", macro_f1)}


def standardize(x_fit: np.ndarray, *others: np.ndarray) -> List[np.ndarray]:
    """Scale every array with the fitting split's mean and std; returns float32.

    Mean and std are computed in float64. Constant coordinates (e.g. layer 0
    [CLS], which is the same vector for every sentence) are centred but not
    scaled, with the same zero-std rule as sklearn's StandardScaler.
    """
    mean = x_fit.mean(axis=0, dtype=np.float64)
    std = x_fit.std(axis=0, dtype=np.float64)
    std[std < 10 * np.finfo(np.float64).eps] = 1.0
    return [((x - mean) / std).astype(np.float32) for x in (x_fit,) + others]


class LinearProbe:
    """Multinomial logistic regression with sklearn's lbfgs objective, solved with PyTorch.

    Minimises (1/n) * [ sum_i cross_entropy_i + ||W||^2 / (2C) ], the objective of
    sklearn's LogisticRegression(solver="lbfgs", penalty="l2") divided by C*n
    (same minimiser; the intercept is not penalised, as in sklearn). Stops when the
    largest gradient entry is below tol, sklearn's lbfgs "gtol" rule, or after
    max_iter iterations. Starts from zero every time (no warm start).
    Full-batch L-BFGS uses every torch thread, so it is several times faster than
    sklearn here, whose multinomial loss runs on one thread; test_probe.py checks
    that both give the same model.
    """

    def __init__(self, C: float, max_iter: int = PROBE_MAX_ITER, tol: float = 1e-4):
        self.C, self.max_iter, self.tol = C, max_iter, tol

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        import torch

        self.classes_, y_index = np.unique(y, return_inverse=True)
        features = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        target = torch.from_numpy(y_index.astype(np.int64))
        n, d = features.shape
        weight = torch.zeros(d, len(self.classes_), requires_grad=True)
        bias = torch.zeros(len(self.classes_), requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weight, bias], lr=1, max_iter=self.max_iter, tolerance_grad=self.tol,
            tolerance_change=64 * np.finfo(np.float64).eps, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = (torch.nn.functional.cross_entropy(features @ weight + bias, target, reduction="sum")
                    + (weight * weight).sum() / (2 * self.C)) / n
            loss.backward()
            return loss

        optimizer.step(closure)
        self.n_iter_ = int(optimizer.state[optimizer._params[0]]["n_iter"])
        self.coef_ = weight.detach().numpy().T.copy()   # (classes, features), as in sklearn
        self.intercept_ = bias.detach().numpy().copy()
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.coef_.T + self.intercept_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


def fit_and_select(
    x_fit: np.ndarray, y_fit: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    x_test: np.ndarray,
    select_metric,
    c_grid: Sequence[float] = C_GRID,
    threads: Optional[int] = None,
) -> Dict[str, object]:
    """Fit one probe per C, choose C on validation, return the chosen model's test predictions."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    x_fit, x_val, x_test = standardize(x_fit, x_val, x_test)
    grid = []
    for c in sorted(c_grid):
        start = time.perf_counter()
        classifier = LinearProbe(C=c).fit(x_fit, y_fit)
        grid.append({
            "C": c,
            "val_score": select_metric(y_val, classifier.predict(x_val)),
            "n_iter": classifier.n_iter_,
            "converged": classifier.n_iter_ < PROBE_MAX_ITER,
            "fit_seconds": round(time.perf_counter() - start, 2),
            "test_pred": classifier.predict(x_test),
        })
    best = max(grid, key=lambda g: (g["val_score"], -g["C"]))  # ties -> smaller C
    return {
        "C": best["C"],
        "val_score": best["val_score"],
        "n_iter": best["n_iter"],
        "converged": best["converged"],
        "at_grid_edge": best["C"] in (min(c_grid), max(c_grid)),
        "test_pred": best["test_pred"],
        "grid": [{k: v for k, v in g.items() if k != "test_pred"} for g in grid],
        "fit_seconds": round(sum(g["fit_seconds"] for g in grid), 2),
    }


def validation_split(labels: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified 90/10 split of row indices, fixed by the probe seed."""
    from sklearn.model_selection import train_test_split

    fit_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=VAL_FRACTION, stratify=labels, random_state=seed)
    return np.sort(fit_idx), np.sort(val_idx)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class Features:
    """Memory-mapped access to one model's feature folder."""

    def __init__(self, folder: Path):
        self.folder = folder
        meta_path = folder / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} not found; run extract_features.py --name {folder.name} first")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def require(self, parts: Iterable[str]) -> None:
        missing = [p for p in parts if p not in self.meta.get("completed", {})]
        if missing:
            raise FileNotFoundError(f"{self.folder}: parts not extracted yet: {missing}")

    def block(self, part: str, layer: int, cell: Cell, rows: Optional[np.ndarray] = None) -> np.ndarray:
        array = np.load(self.folder / f"{part}_layer{layer}.npy", mmap_mode="r")
        block = array[:, cell.start:cell.stop] if rows is None else array[rows, cell.start:cell.stop]
        return np.ascontiguousarray(block, dtype=np.float32)

    def labels(self, name: str) -> np.ndarray:
        return np.load(self.folder / f"{name}.npy")


def task_jobs(task: str, seeds: Sequence[int]) -> List[Tuple[Cell, int]]:
    # POS has a fixed dev split and a deterministic probe, so one "seed" (0) suffices.
    task_seeds = (0,) if task == "pos" else tuple(seeds)
    return [(cell, seed) for cell in probe_cells() for seed in task_seeds]


def run_job(features: Features, task: str, cell: Cell, seed: int, c_grid, threads: int) -> Dict[str, object]:
    metric_name, metric = SELECTION_METRIC[task]
    if task == "pos":
        x_fit = features.block("ud_train", cell.layer, cell)
        y_fit = features.labels("ud_train_upos")
        x_val = features.block("ud_dev", cell.layer, cell)
        y_val = features.labels("ud_dev_upos")
        x_test = features.block("ud_test", cell.layer, cell)
        y_test = features.labels("ud_test_upos")
    else:
        y_all = features.labels("tweet_train_labels")
        fit_idx, val_idx = validation_split(y_all, seed)
        x_fit = features.block("tweet_train", cell.layer, cell, fit_idx)
        x_val = features.block("tweet_train", cell.layer, cell, val_idx)
        y_fit, y_val = y_all[fit_idx], y_all[val_idx]
        x_test = features.block("tweet_test", cell.layer, cell)
        y_test = features.labels("tweet_test_labels")

    result = fit_and_select(x_fit, y_fit, x_val, y_val, x_test, metric, c_grid, threads)
    pred = result.pop("test_pred")
    row = {
        "task": task, **asdict(cell), "probe_seed": seed,
        "n_fit": len(y_fit), "n_val": len(y_val), "n_test": len(y_test),
        "selection_metric": metric_name,
        **{k: v for k, v in result.items() if k != "grid"},
        "test_accuracy": accuracy(y_test, pred),
        "test_macro_f1": macro_f1(y_test, pred),
    }
    return {"row": row, "grid": result["grid"], "pred": pred.astype(np.int8)}


# ---------------------------------------------------------------------------
# Progress and output
# ---------------------------------------------------------------------------

def _atomic_savez(path: Path, **arrays) -> None:
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def save_progress(folder: Path, key: str, job: Dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_savez(folder / f"{key}.npz", pred=job["pred"],
                  info=np.array(json.dumps({"row": job["row"], "grid": job["grid"]})))


def load_progress(folder: Path, key: str) -> Optional[Dict[str, object]]:
    path = folder / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        info = json.loads(str(stored["info"]))
        return {"row": info["row"], "grid": info["grid"], "pred": stored["pred"]}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the syntax-preservation POS and sentiment probes for one model.")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base or mipic_s42.")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--threads", type=int, default=None,
                        help="torch threads per fit (default: torch's choice, normally the physical cores). "
                             "Changes only float rounding; recorded in the output.")
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=list(PROBE_SEEDS))
    args = parser.parse_args(argv)

    features = Features(args.features_dir / args.name)
    parts = {"pos": ["ud_train", "ud_dev", "ud_test"], "sentiment": ["tweet_train", "tweet_test"]}
    features.require([p for t in args.tasks for p in parts[t]])
    progress_dir = features.folder / "probe_progress"
    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    threads = torch.get_num_threads()
    # Thread count is not a setting: it changes only float rounding (same test
    # accuracy at 12 and 16 threads on base), so a run may resume with another one.
    settings = {"c_grid": sorted(args.c_grid), "probe_seeds": sorted(args.probe_seeds),
                "val_fraction": VAL_FRACTION}
    settings_path = progress_dir / "settings.json"
    if settings_path.exists():
        stored = json.loads(settings_path.read_text(encoding="utf-8"))
        # Progress written before the thread count stopped being a setting stored it here.
        legacy_threads = stored.pop("threads_of_earlier_rows", stored.pop("threads", None))
        if stored != settings:
            raise ValueError(f"{progress_dir} holds results from other settings; delete it to start over")
        if legacy_threads is not None:
            settings["threads_of_earlier_rows"] = legacy_threads
    progress_dir.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings), encoding="utf-8")

    jobs = [(task, cell, seed) for task in args.tasks for cell, seed in task_jobs(task, args.probe_seeds)]
    print(f"{args.name}: {len(jobs)} probes x {len(args.c_grid)} values of C, {threads} torch threads", flush=True)
    done: Dict[str, Dict[str, object]] = {}
    start = time.perf_counter()
    for number, (task, cell, seed) in enumerate(jobs, start=1):
        key = prediction_key(task, cell, seed)
        job = load_progress(progress_dir, key)
        if job is None:
            job = run_job(features, task, cell, seed, args.c_grid, threads)
            job["row"]["threads"] = threads
            save_progress(progress_dir, key, job)
            row = job["row"]
            flags = ("" if row["converged"] else "  NOT CONVERGED") + ("  C AT GRID EDGE" if row["at_grid_edge"] else "")
            print(f"  [{number}/{len(jobs)}] {key}: C={row['C']:g}, val {100 * row['val_score']:.2f}, "
                  f"test acc {100 * row['test_accuracy']:.2f}, macro-F1 {100 * row['test_macro_f1']:.2f} "
                  f"({row['fit_seconds']:.0f}s){flags}", flush=True)
        job["row"].setdefault("threads", settings.get("threads_of_earlier_rows"))
        done[key] = job

    rows, grid_rows, preds = [], [], {}
    for key, job in done.items():
        rows.append({"model": args.name, **job["row"]})
        for g in job["grid"]:
            grid_rows.append({"model": args.name, "key": key, **g})
        preds[key] = job["pred"]
    if "pos" in args.tasks:
        preds["pos_labels"] = features.labels("ud_test_upos")
    if "sentiment" in args.tasks:
        preds["sentiment_labels"] = features.labels("tweet_test_labels")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / f"{args.name}_probe.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(args.out_dir / f"{args.name}_cgrid.csv", index=False)
    _atomic_savez(args.out_dir / f"{args.name}_preds.npz", **preds)

    for task in args.tasks:
        metric = "test_accuracy" if task == "pos" else "test_macro_f1"
        view = table[table["task"] == task].groupby(["layer", "coords"])[metric].mean().unstack() * 100
        print(f"\n{task}: test {metric.replace('test_', '')} (%), mean over probe seeds")
        print(view.reindex(columns=["full", "prefix", "suffix"]).round(2).to_string())
    print(f"\n{len(done)} probes, {time.perf_counter() - start:.0f}s this run. "
          f"Saved {args.name}_probe.csv, {args.name}_cgrid.csv and {args.name}_preds.npz to {args.out_dir}")


if __name__ == "__main__":
    main()
