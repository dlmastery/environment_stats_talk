"""AFTER — split (inductive) conformal prediction intervals.

Conformal prediction turns *any* point forecaster into one with a finite-sample,
distribution-free **marginal coverage guarantee**. We use the split / inductive
recipe: hold out a calibration set the model never trained on, score each
calibration point with a *nonconformity* score, and read the band off an empirical
quantile of those scores. Because the calibration scores and a future test score
are exchangeable, the resulting interval covers a fresh point with probability at
least the nominal level — no Gaussianity, no homoscedasticity assumption.

Two variants are implemented:

1. ``split_conformal`` — **absolute-residual** score s_i = |y_i - yhat(x_i)|.
   The band is yhat(x) ± Q, where Q is the (1-alpha)(1 + 1/n_cal)-empirical
   quantile of the calibration scores. Constant width, but the width is calibrated
   to the *true* residual distribution (skew, heavy tails and all) rather than to a
   Gaussian sigma. This already fixes marginal coverage.

2. ``normalized_conformal`` (locally-adaptive) — score s_i = |y_i - yhat(x_i)| /
   sigma_hat(x_i), where sigma_hat(x) is a learned estimate of the conditional
   residual spread (``forecaster.fit_residual_scale_model``). The band is
   yhat(x) ± Q * sigma_hat(x): **variable width** that widens where residuals are
   large (e.g. the warm season) and tightens where they are small, while keeping
   the same marginal-coverage guarantee. This targets *conditional* coverage that
   the constant-width band cannot.

Finite-sample quantile level
----------------------------
For a calibration set of size n and miscoverage alpha = 1 - level, the conformal
quantile is taken at rank ceil((n+1)(1-alpha)) / n (equivalently the
(1-alpha)(1+1/n) empirical quantile). The +1 is the finite-sample correction that
delivers the >= level guarantee for exchangeable data.

Caveat (stated, not hidden): the guarantee is **marginal** coverage averaged over
the joint distribution. It is also exact only under *exchangeability*; forecast
residuals are weakly serially dependent, so we use a calibration block adjacent to
the test block and report the *empirical* test coverage to confirm the guarantee
holds in practice. Conditional (per-region / per-season) coverage is not guaranteed
by either variant; the normalized variant only *improves* it. See README.

This module wraps the *shared* ridge point forecaster (``forecaster.py``). Run from
the REPO ROOT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXP_DIR = Path(__file__).resolve().parents[1]
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from forecaster import (  # noqa: E402
    build_lag_features, fit_point_forecaster, fit_residual_scale_model,
    DEFAULT_LOOKBACK,
)

__all__ = [
    "conformal_quantile",
    "split_conformal",
    "normalized_conformal",
    "run_after",
    "SIGMA_FLOOR",
]

# Strictly-positive floor for the conditional spread estimate, as a fraction of the
# mean absolute calibration residual, so the normaliser never collapses to ~0.
SIGMA_FLOOR_FRAC = 0.1


def conformal_quantile(scores, level: float) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores``.

    Returns the value Q such that, with the +1/n correction, a fresh exchangeable
    score is <= Q with probability >= ``level``. Implemented as the rank
    ceil((n+1)*level)/n empirical quantile (clipped to n -> +inf when the level is
    so high the correction exceeds the sample, the honest "infinite band" case).
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * level))
    if rank > n:
        return float("inf")  # not enough calibration points for this level
    # rank-th smallest score (1-indexed) -> index rank-1 of the sorted scores.
    return float(np.sort(scores)[rank - 1])


def split_conformal(point_cal, y_cal, point_test, level: float):
    """Split conformal with the absolute-residual score (constant-width band).

    Returns (lower, upper, half_width, Q). ``half_width`` is the constant Q,
    broadcast to the length of ``point_test``.
    """
    point_cal = np.asarray(point_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    point_test = np.asarray(point_test, dtype=float)

    scores = np.abs(y_cal - point_cal)
    Q = conformal_quantile(scores, level)
    lower = point_test - Q
    upper = point_test + Q
    return lower, upper, np.full_like(point_test, Q), Q


def normalized_conformal(point_cal, y_cal, sigma_cal, point_test, sigma_test,
                         level: float):
    """Locally-adaptive (normalized) split conformal.

    Score s_i = |y_i - yhat_i| / sigma_hat(x_i). Band is yhat ± Q * sigma_hat(x),
    so the width varies with the conditional-spread estimate ``sigma_test``.
    Returns (lower, upper, half_width, Q) where ``half_width = Q * sigma_test``.
    """
    point_cal = np.asarray(point_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    sigma_cal = np.asarray(sigma_cal, dtype=float)
    point_test = np.asarray(point_test, dtype=float)
    sigma_test = np.asarray(sigma_test, dtype=float)

    scores = np.abs(y_cal - point_cal) / sigma_cal
    Q = conformal_quantile(scores, level)
    half = Q * sigma_test
    lower = point_test - half
    upper = point_test + half
    return lower, upper, half, Q


def run_after(train_series, cal_series, test_series, levels=(0.80, 0.90, 0.95),
              lookback: int = DEFAULT_LOOKBACK, alpha: float = 1.0) -> dict:
    """Fit the shared point forecaster on TRAIN, calibrate on a held-out CAL set.

    Pipeline (no leakage): the point forecaster and the conditional-spread model are
    fit on TRAIN only; conformal quantiles are read from CAL only; coverage/width
    are measured on TEST only. Returns both the constant-width split-conformal band
    and the variable-width normalized band per level.
    """
    X_tr, y_tr, _ = build_lag_features(train_series, lookback)
    X_cal, y_cal, _ = build_lag_features(cal_series, lookback)
    X_te, y_te, te_index = build_lag_features(test_series, lookback)

    model = fit_point_forecaster(X_tr, y_tr, alpha=alpha)
    pred_tr = model.predict(X_tr)
    pred_cal = model.predict(X_cal)
    pred_te = model.predict(X_te)

    # Conditional-spread model for the normalized variant (fit on TRAIN residuals).
    abs_resid_tr = np.abs(y_tr - pred_tr)
    scale_model = fit_residual_scale_model(X_tr, abs_resid_tr, alpha=alpha)
    sigma_floor = SIGMA_FLOOR_FRAC * float(np.mean(abs_resid_tr) + 1e-12)
    sigma_cal = np.clip(scale_model.predict(X_cal), sigma_floor, None)
    sigma_te = np.clip(scale_model.predict(X_te), sigma_floor, None)

    split_bands, norm_bands = {}, {}
    for lvl in levels:
        lo, hi, half, Q = split_conformal(pred_cal, y_cal, pred_te, lvl)
        split_bands[f"{lvl:.2f}"] = {
            "lower": lo, "upper": hi, "half_width": half, "Q": Q,
        }
        nlo, nhi, nhalf, nQ = normalized_conformal(
            pred_cal, y_cal, sigma_cal, pred_te, sigma_te, lvl)
        norm_bands[f"{lvl:.2f}"] = {
            "lower": nlo, "upper": nhi, "half_width": nhalf, "Q": nQ,
        }

    return {
        "method": "split_conformal",
        "y_true": y_te,
        "point_pred": pred_te,
        "test_index": te_index,
        "sigma_test": sigma_te,
        "sigma_floor": sigma_floor,
        "n_cal": int(len(y_cal)),
        "bands": split_bands,            # constant-width split conformal
        "normalized_bands": norm_bands,  # variable-width locally-adaptive conformal
        "lookback": lookback,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import daily_temperature

    df = daily_temperature(n_years=10, seed=0)
    n = len(df)
    tr = df.iloc[: int(0.6 * n)]["t2m"]
    cal = df.iloc[int(0.6 * n): int(0.8 * n)]["t2m"]
    te = df.iloc[int(0.8 * n):]["t2m"]
    out = run_after(tr, cal, te)
    for lvl in ("0.80", "0.90", "0.95"):
        b = out["bands"][lvl]
        nb = out["normalized_bands"][lvl]
        cov = float(np.mean((out["y_true"] >= b["lower"]) & (out["y_true"] <= b["upper"])))
        ncov = float(np.mean((out["y_true"] >= nb["lower"]) & (out["y_true"] <= nb["upper"])))
        print(f"level={lvl}  split_cov={cov:.3f} (w={2*b['half_width'][0]:.2f})  "
              f"norm_cov={ncov:.3f} (mean w={2*np.mean(nb['half_width']):.2f})")
