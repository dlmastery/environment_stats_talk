"""BEFORE — Gaussian / normal-theory prediction intervals.

The interval an environmental statistician reaches for *before* anything fancier:
assume the forecast residuals are i.i.d. Normal with constant variance, estimate
that variance from the training residuals, and put a symmetric band

    yhat(x)  ±  z(level) * sigma_hat

around every point forecast, where ``sigma_hat`` is the (single) training residual
standard deviation and ``z(level)`` is the standard-normal quantile for the desired
central coverage (1.6449 for 90%, 1.96 for 95%, 1.2816 for 80%).

Why this is the natural BEFORE method
-------------------------------------
It is the textbook prediction interval: closed-form, one number of "spread" for the
whole series, no calibration set. It is *correct* when residuals really are
homoscedastic Gaussian. The point of the experiment is that environmental residuals
usually are **not**: daily-temperature residuals here are heteroscedastic (wider in
the warm season, where sporadic heat extremes live) and right-skewed. A single
Gaussian sigma then mis-states the spread, so the empirical coverage of these bands
drifts away from the nominal level — over- or under-covering depending on the level.

This module wraps the *shared* ridge point forecaster (``forecaster.py``); only the
interval construction is "BEFORE". Run from the REPO ROOT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from forecaster import (  # noqa: E402
    build_lag_features, fit_point_forecaster, DEFAULT_LOOKBACK,
)

__all__ = ["normal_z", "normal_theory_intervals", "run_before"]


def normal_z(level: float) -> float:
    """Standard-normal two-sided quantile z such that P(|Z| <= z) = level.

    Uses scipy if available, else a high-accuracy rational approximation of the
    inverse normal CDF (Acklam) so the module has no hard scipy dependency.
    """
    p = 0.5 + level / 2.0  # upper tail point for a central `level` interval
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:  # pragma: no cover - scipy is in requirements, fallback only
        return float(_norm_ppf(p))


def _norm_ppf(p: float) -> float:
    """Acklam's inverse-normal-CDF approximation (abs error < ~1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def normal_theory_intervals(point_pred, sigma_hat: float, level: float):
    """Symmetric Gaussian band yhat ± z(level)*sigma_hat.

    Returns (lower, upper, half_width) as 1-D arrays; ``half_width`` is the
    constant ``z*sigma_hat`` broadcast to the length of ``point_pred``.
    """
    point_pred = np.asarray(point_pred, dtype=float)
    z = normal_z(level)
    half = z * float(sigma_hat)
    lower = point_pred - half
    upper = point_pred + half
    return lower, upper, np.full_like(point_pred, half)


def run_before(train_series, test_series, levels=(0.80, 0.90, 0.95),
               lookback: int = DEFAULT_LOOKBACK, alpha: float = 1.0) -> dict:
    """Fit the shared point forecaster on TRAIN, then build normal-theory PIs.

    The "spread" is the standard deviation of the *in-sample training residuals*
    — the only thing the textbook recipe gives you without a separate calibration
    set. Returns a dict with the point forecast on the test set, the per-level
    bands, and the metadata the orchestrator needs to compute coverage/width.
    """
    X_tr, y_tr, _ = build_lag_features(train_series, lookback)
    X_te, y_te, te_index = build_lag_features(test_series, lookback)

    model = fit_point_forecaster(X_tr, y_tr, alpha=alpha)
    resid_tr = y_tr - model.predict(X_tr)
    sigma_hat = float(np.std(resid_tr, ddof=1))

    pred_te = model.predict(X_te)
    bands = {}
    for lvl in levels:
        lo, hi, half = normal_theory_intervals(pred_te, sigma_hat, lvl)
        bands[f"{lvl:.2f}"] = {"lower": lo, "upper": hi, "half_width": half}

    return {
        "method": "normal_theory",
        "sigma_hat": sigma_hat,
        "y_true": y_te,
        "point_pred": pred_te,
        "test_index": te_index,
        "bands": bands,
        "lookback": lookback,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import daily_temperature, time_split

    df = daily_temperature(n_years=8, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    out = run_before(tr["t2m"], te["t2m"])
    print(f"sigma_hat={out['sigma_hat']:.3f}")
    for lvl, b in out["bands"].items():
        cov = float(np.mean((out["y_true"] >= b["lower"]) & (out["y_true"] <= b["upper"])))
        print(f"  level={lvl} empirical_coverage={cov:.3f} half_width={b['half_width'][0]:.3f}")
