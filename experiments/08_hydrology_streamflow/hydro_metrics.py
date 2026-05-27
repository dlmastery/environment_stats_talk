"""Hydrology forecast-skill metrics: NSE, KGE, RMSE.

These are the standard streamflow-evaluation metrics used throughout rainfall-runoff
modelling (and in the published LSTM-rainfall-runoff studies). Plain NumPy so they
run anywhere. RMSE re-uses the convention from ``common.metrics`` but is re-defined
here so this module is self-contained for both the BEFORE and AFTER scripts.

- **NSE** — Nash-Sutcliffe Efficiency:  1 - SS_res / SS_tot.  Range (-inf, 1];
  1 = perfect, 0 = no better than predicting the observed mean, < 0 = worse than
  the mean. The headline streamflow skill score; "good" models are NSE > ~0.5-0.7.
- **KGE** — Kling-Gupta Efficiency (Gupta et al. 2009): combines correlation (r),
  the bias ratio (beta = mean_sim/mean_obs) and the variability ratio
  (alpha = std_sim/std_obs) as  1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2).
  Range (-inf, 1]; 1 = perfect. KGE -0.41 is the "mean-flow benchmark" threshold.
- **RMSE** — root mean squared error (mm/day); lower is better.
"""
from __future__ import annotations

import numpy as np

__all__ = ["nse", "kge", "rmse", "all_metrics"]


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (mm/day); lower is better."""
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency. 1 = perfect, 0 = as good as the mean, <0 = worse."""
    yt, yp = _arr(y_true), _arr(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def kge(y_true, y_pred) -> float:
    """Kling-Gupta Efficiency (2009 form). 1 = perfect."""
    yt, yp = _arr(y_true), _arr(y_pred)
    mu_t, mu_p = np.mean(yt), np.mean(yp)
    sd_t, sd_p = np.std(yt), np.std(yp)
    if sd_t == 0 or mu_t == 0:
        return float("nan")
    # Pearson correlation
    if sd_p == 0:
        r = 0.0
    else:
        r = float(np.corrcoef(yt, yp)[0, 1])
        if not np.isfinite(r):
            r = 0.0
    alpha = sd_p / sd_t          # variability ratio
    beta = mu_p / mu_t           # bias ratio
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def all_metrics(y_true, y_pred) -> dict:
    """Convenience: return {'rmse','nse','kge'} for a prediction."""
    return {"rmse": rmse(y_true, y_pred), "nse": nse(y_true, y_pred),
            "kge": kge(y_true, y_pred)}
