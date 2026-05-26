"""Shared metrics for environmental-statistics experiments.

Plain NumPy implementations so they run anywhere (no sklearn dependency required,
though sklearn is available). Every metric is documented with the convention used
in environmental forecasting / environmetrics.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rmse", "mae", "bias", "mape", "r2",
    "anomaly_correlation", "latitude_weighted_rmse",
    "skill_score", "classification_report_simple",
]


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (lower is better)."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error (lower is better)."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def bias(y_true, y_pred) -> float:
    """Mean error (pred - true); sign indicates over/under-forecast."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(yp - yt))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%) — guard against divide-by-zero."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (higher is better, max 1)."""
    yt, yp = _arr(y_true), _arr(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def anomaly_correlation(y_true, y_pred, climatology=None) -> float:
    """Anomaly Correlation Coefficient (ACC), the standard skill metric in NWP.

    ACC correlates forecast anomalies with observed anomalies relative to a
    climatology (defaults to the mean of ``y_true``). Range [-1, 1]; higher is
    better; operational forecasts target ACC > 0.6.
    """
    yt, yp = _arr(y_true), _arr(y_pred)
    clim = np.mean(yt) if climatology is None else float(climatology)
    at, ap = yt - clim, yp - clim
    denom = np.sqrt(np.sum(at ** 2) * np.sum(ap ** 2))
    return float(np.sum(at * ap) / denom) if denom > 0 else float("nan")


def latitude_weighted_rmse(y_true, y_pred, lats) -> float:
    """Latitude-weighted RMSE for gridded fields (weights = cos(lat)).

    This is the convention used by ERA5-based weather-model benchmarks: grid
    cells near the poles cover less area, so they are down-weighted by cos(lat).
    ``y_true``/``y_pred`` have shape (..., n_lat, n_lon); ``lats`` is length n_lat
    in degrees.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    lats = np.asarray(lats, dtype=float)
    w = np.cos(np.deg2rad(lats))
    w = w / np.mean(w)
    w = w.reshape((1,) * (yt.ndim - 2) + (len(lats), 1))
    se = (yt - yp) ** 2 * w
    return float(np.sqrt(np.mean(se)))


def skill_score(metric_model: float, metric_reference: float,
                higher_is_better: bool = False) -> float:
    """Skill score of a model vs a reference baseline.

    For error metrics (lower better): 1 - model/reference  (1 = perfect, 0 = no
    improvement, <0 = worse than baseline). For score metrics (higher better):
    (model - reference) / (perfect - reference) is context-specific, so here we
    return the relative improvement (model - reference)/|reference|.
    """
    if higher_is_better:
        return float((metric_model - metric_reference) / (abs(metric_reference) + 1e-12))
    return float(1.0 - metric_model / (metric_reference + 1e-12))


def classification_report_simple(y_true, y_pred) -> dict:
    """Accuracy + macro precision/recall/F1 without sklearn (multiclass)."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([yt, yp]))
    accuracy = float(np.mean(yt == yp))
    precs, recs, f1s = [], [], []
    for c in classes:
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "n_classes": int(len(classes)),
    }
