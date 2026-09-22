from __future__ import annotations

import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _pearson_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_centered = y_true - np.mean(y_true)
    pred_centered = y_pred - np.mean(y_pred)
    denominator = np.sum(true_centered**2) * np.sum(pred_centered**2)
    if denominator == 0:
        return float("nan")
    numerator = np.sum(true_centered * pred_centered) ** 2
    return float(numerator / denominator)


def _zero_intercept_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum(y_pred**2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0 or total == 0:
        return float("nan")
    k = np.sum(y_true * y_pred) / denominator
    residual = np.sum((y_true - k * y_pred) ** 2)
    return float(1 - residual / total)


def rm2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """DTA-style r_m^2 metric used by DeepDTA-style benchmarks."""
    r2 = _pearson_r2(y_true, y_pred)
    r02 = _zero_intercept_r2(y_true, y_pred)
    if np.isnan(r2) or np.isnan(r02):
        return float("nan")
    return float(r2 * (1 - np.sqrt(abs((r2 * r2) - (r02 * r02)))))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, ci_higher_is_better: bool = True) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    ci_scores = y_pred if ci_higher_is_better else -y_pred
    try:
        ci = float(concordance_index(y_true, ci_scores))
    except ZeroDivisionError:
        ci = float("nan")

    metrics = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "rm2": rm2_score(y_true, y_pred),
        "ci": ci,
    }
    metrics["pearson"] = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    metrics["spearman"] = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 1 else float("nan")
    return metrics
