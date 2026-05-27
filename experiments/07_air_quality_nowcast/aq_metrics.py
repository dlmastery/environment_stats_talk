"""Air-quality forecast-skill metrics: RMSE, MAE, skill-vs-persistence, spike F1.

Plain NumPy so they run anywhere. Self-contained for both the BEFORE and AFTER
scripts (RMSE/MAE re-use the convention from ``common.metrics`` but are re-defined
here so this module stands alone).

Metrics
-------
- **RMSE / MAE** — error of the PM2.5 prediction (micrograms/m^3); lower is better.
- **Skill vs persistence** — ``1 - RMSE_model / RMSE_persistence``. Persistence
  (predict the last observed value) is the honest "do nothing" reference for any
  nowcast. 1 = perfect, 0 = no better than persistence, < 0 = worse. This is the
  number that matters: beating persistence is the bar a forecast must clear.
- **Spike / exceedance F1** — air-quality decisions are about *exceedances* (when
  does PM2.5 cross an "unhealthy" threshold?). We binarize observed and predicted
  PM2.5 at a shared threshold and report precision / recall / F1 for the positive
  (exceedance) class. F1 captures both missing episodes (recall) and false alarms
  (precision); a forecast that nails the mean but smears out the spikes scores poorly.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "rmse", "mae", "skill_vs_persistence",
    "exceedance_labels", "spike_detection_scores", "all_metrics",
]


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (micrograms/m^3); lower is better."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error (micrograms/m^3); lower is better."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def skill_vs_persistence(rmse_model: float, rmse_persistence: float) -> float:
    """Skill score against the persistence reference: ``1 - model/persistence``.

    1 = perfect, 0 = ties persistence, < 0 = worse than persistence.
    """
    return float(1.0 - rmse_model / (rmse_persistence + 1e-12))


def exceedance_labels(values, threshold: float) -> np.ndarray:
    """Binarize a PM2.5 series at ``threshold`` (1 = exceedance / spike)."""
    return (_arr(values) > float(threshold)).astype(int)


def spike_detection_scores(y_true, y_pred, threshold: float) -> dict:
    """Precision / recall / F1 for the exceedance (spike) class.

    Both ``y_true`` and ``y_pred`` are PM2.5 series; they are binarized at the SAME
    ``threshold`` (defined on the observations) so model and obs are judged on one
    bar. Returns a dict with precision, recall, f1, the threshold, support (number
    of true exceedances), and the confusion counts. F1 is always finite (0 when
    there are no true and no predicted exceedances).
    """
    yt = exceedance_labels(y_true, threshold)
    yp = exceedance_labels(y_pred, threshold)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "support_true": int(tp + fn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def all_metrics(y_true, y_pred, rmse_persistence: float, threshold: float) -> dict:
    """Convenience bundle: RMSE, MAE, skill-vs-persistence, and spike F1.

    ``rmse_persistence`` is the persistence baseline's RMSE on the SAME targets
    (so skill is comparable across methods); ``threshold`` defines exceedances.
    """
    r = rmse(y_true, y_pred)
    spike = spike_detection_scores(y_true, y_pred, threshold)
    return {
        "rmse": r,
        "mae": mae(y_true, y_pred),
        "skill_vs_persistence": skill_vs_persistence(r, rmse_persistence),
        "spike_precision": spike["precision"],
        "spike_recall": spike["recall"],
        "spike_f1": spike["f1"],
        "spike_support": spike["support_true"],
    }
