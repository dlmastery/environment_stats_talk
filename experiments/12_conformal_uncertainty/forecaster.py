"""Shared point forecaster for Experiment 12 (conformal uncertainty).

A deliberately *light*, CPU-only point forecaster for daily 2 m temperature: a
ridge regression on lag features plus a sin/cos day-of-year seasonal encoding.
Both the BEFORE (normal-theory) and AFTER (conformal) interval methods wrap this
*same* point forecaster, so the only thing that differs between the two stories is
how the prediction interval is constructed — never the point prediction. That is
what makes the calibration comparison fair.

Why ridge-on-lags (and not the LSTM from Exp01)?
- The interval methods are the subject here, not the regressor. A transparent,
  fast, deterministic regressor keeps the experiment runs-anywhere (no GPU, no
  torch) and keeps attention on coverage rather than point skill.
- Lag features + day-of-year capture the AR(1) persistence and the seasonal cycle
  of the synthetic series, leaving residuals that are *heteroscedastic and
  right-skewed* (warm-season heat extremes) — exactly the regime where a single
  Gaussian residual-sigma interval is miscalibrated and conformal helps.

Contract used by before/ and after/:

    build_lag_features(series, lookback)         -> (X, y, index)
    fit_point_forecaster(X_tr, y_tr, alpha)      -> fitted model
    model.predict(X)                             -> point predictions
    residual_scale_features(X)                   -> features for the conditional
                                                     residual-spread model (AFTER's
                                                     locally-adaptive variant)

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sklearn.linear_model import Ridge  # noqa: E402

__all__ = [
    "build_lag_features",
    "fit_point_forecaster",
    "fit_residual_scale_model",
    "DEFAULT_LOOKBACK",
]

DEFAULT_LOOKBACK = 7


def _as_series(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy().ravel().astype(float)
    return np.asarray(x, dtype=float).ravel()


def build_lag_features(series, lookback: int = DEFAULT_LOOKBACK, index=None):
    """Build a 1-step-ahead supervised dataset from a daily series.

    Features for predicting ``y(t)`` are the previous ``lookback`` values
    ``y(t-1)..y(t-lookback)`` plus a sin/cos day-of-year seasonal encoding for the
    target day ``t``. No future information enters any feature (no leakage).

    Returns (X, y, target_index) where target_index aligns with ``y``. If a pandas
    object with a DatetimeIndex is passed (or ``index`` given), the day-of-year is
    taken from it; otherwise a synthetic day-of-year (t mod 365) is used.
    """
    y_full = _as_series(series)
    n = len(y_full)
    if isinstance(series, (pd.Series, pd.DataFrame)) and index is None:
        idx = series.index
    else:
        idx = index

    if isinstance(idx, pd.DatetimeIndex):
        doy_full = idx.dayofyear.to_numpy().astype(float)
        target_index_full = idx
    else:
        doy_full = ((np.arange(n)) % 365 + 1).astype(float)
        target_index_full = np.arange(n)

    rows_X, rows_y, rows_idx = [], [], []
    for t in range(lookback, n):
        lags = y_full[t - lookback:t][::-1]  # y(t-1), y(t-2), ..., y(t-lookback)
        doy = doy_full[t]
        sin = np.sin(2 * np.pi * doy / 365.25)
        cos = np.cos(2 * np.pi * doy / 365.25)
        rows_X.append(np.concatenate([lags, [sin, cos]]))
        rows_y.append(y_full[t])
        rows_idx.append(target_index_full[t])

    X = np.asarray(rows_X, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    if isinstance(idx, pd.DatetimeIndex):
        target_index = pd.DatetimeIndex(rows_idx)
    else:
        target_index = np.asarray(rows_idx)
    return X, y, target_index


def fit_point_forecaster(X_tr, y_tr, alpha: float = 1.0) -> Ridge:
    """Fit a ridge regression point forecaster (the shared regressor).

    Ridge (L2-penalised least squares) is transparent, deterministic and fast.
    ``alpha`` is the L2 strength; the default is a light regularisation that keeps
    the lag coefficients stable without over-smoothing the AR(1) persistence.
    """
    model = Ridge(alpha=alpha)
    model.fit(np.asarray(X_tr, dtype=float), np.asarray(y_tr, dtype=float))
    return model


def fit_residual_scale_model(X_tr, abs_resid_tr, alpha: float = 1.0) -> Ridge:
    """Fit a model of the *conditional residual spread* (for locally-adaptive CP).

    Locally-adaptive (normalised) conformal needs an estimate sigma_hat(x) of how
    large the residual tends to be at input ``x``. We regress the absolute training
    residuals on the same features with ridge, then clip to a small positive floor
    at prediction time so the normaliser is always strictly positive. This is the
    standard "normalising function" used by normalised/locally-adaptive conformal
    regression — it lets interval *width* vary with the inputs (e.g. wider in the
    warm season where heat extremes inflate the spread).
    """
    model = Ridge(alpha=alpha)
    model.fit(np.asarray(X_tr, dtype=float),
              np.asarray(abs_resid_tr, dtype=float))
    return model
