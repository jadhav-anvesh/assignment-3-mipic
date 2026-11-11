"""Task metrics and statistical confidence reporting."""

from typing import Dict, List, Sequence
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro-averaged F1 score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_spearman(cos_sim: np.ndarray, gold_scores: np.ndarray) -> float:
    """Compute Spearman rank correlation between predicted similarities and gold scores."""
    corr, _ = spearmanr(cos_sim, gold_scores)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_summary_statistics(scores: List[float]) -> Dict[str, float]:
    """Compute mean, standard deviation, and 95% confidence intervals across seeds."""
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0, "count": 0}
    if n == 1:
        return {"mean": scores[0], "std": 0.0, "ci_low": scores[0], "ci_high": scores[0], "count": 1}

    mean_val = float(np.mean(scores))
    std_val = float(np.std(scores, ddof=1))
    sem = std_val / np.sqrt(n)
    # Student-t 95% confidence interval
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = float(mean_val - t_crit * sem)
    ci_high = float(mean_val + t_crit * sem)

    return {
        "mean": mean_val,
        "std": std_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "count": n,
    }

