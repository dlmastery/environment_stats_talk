"""BEFORE -- the traditional air-quality nowcast baselines (PM2.5 history only).

These are the models an environmental statistician reaches for first when asked to
nowcast PM2.5 and they deliberately use **only the PM2.5 history** -- no weather
covariates. That is the whole point of the BEFORE/AFTER contrast: history-only
methods can ride PM2.5's self-persistence partway, but they are blind to the
*driver* of the spikes (a wind drop + a collapsing boundary layer), so they lag and
smear the exceedance episodes. The AFTER model, which sees the weather, closes that
gap.

Three classical references, in increasing sophistication:

1. **Persistence** -- predict the last observed value (``y_hat(t) = y(t-1)``). The
   honest "do nothing" reference; every nowcast must beat it to be worth anything.
2. **Linear (lagged) regression / AR(p)** -- ordinary least squares of PM2.5 on its
   own recent lags ``[y(t-1), ..., y(t-p)]``. A transparent linear autoregression.
3. **ARIMA** -- a Box-Jenkins ARIMA(p,d,q) fit with statsmodels (the canonical
   univariate time-series forecaster). Optional; falls back gracefully to the linear
   model if statsmodels is unavailable or the fit fails.

All three forecast **one step ahead** in a leak-free, walk-forward manner: at test
step ``t`` they may use only observations up to ``t-1`` (the linear model is *fit*
on train only, then applied with true past lags -- a standard one-step nowcast).

Contract (shared with the AFTER side so the orchestrator treats them uniformly):

    run_before(train_df, test_df, ...) -> dict with per-method
        {"rmse":.., "mae":.., "y_true":[...], "y_pred":[...]}

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1]   # experiments/07_air_quality_nowcast
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aq_metrics import rmse, mae  # type: ignore  # noqa: E402

__all__ = [
    "persistence_forecast",
    "build_ar_design",
    "fit_linear_ar",
    "linear_ar_forecast",
    "arima_forecast",
    "run_before",
]

_TARGET = "pm25"


# --------------------------------------------------------------------------- #
# 1) Persistence
# --------------------------------------------------------------------------- #
def persistence_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-step persistence: y_hat(t) = y(t-1), using the true previous observation.

    The first test step uses the last train value as its "previous". Returns
    (y_true, y_pred) over the full test block.
    """
    y_test = test_df[_TARGET].to_numpy().astype(float)
    last_train = float(train_df[_TARGET].to_numpy()[-1])
    y_prev = np.concatenate([[last_train], y_test[:-1]])
    return y_test, np.clip(y_prev, 0.0, None)


# --------------------------------------------------------------------------- #
# 2) Linear autoregression on PM2.5 lags
# --------------------------------------------------------------------------- #
def build_ar_design(series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build an AR design matrix X = [y(t-1), ..., y(t-p)] and target y = y(t).

    Drops the first ``n_lags`` rows (no full history). Returns (X, y) float arrays.
    """
    y = np.asarray(series, dtype=float).ravel()
    n = len(y)
    if n <= n_lags:
        return np.zeros((0, n_lags)), np.zeros((0,))
    cols = [y[n_lags - k - 1: n - k - 1] for k in range(n_lags)]
    X = np.column_stack(cols)            # column k is lag (k+1)
    target = y[n_lags:]
    return X, target


def fit_linear_ar(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    """Ridge-stabilized least-squares AR fit. Returns (w, b)."""
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    coef = np.linalg.solve(A, Xb.T @ y)
    return coef[:-1], coef[-1]


def linear_ar_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       n_lags: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Fit an AR(p) on train PM2.5, then one-step-ahead predict the test block.

    Leak-free: predictions for each test step use the TRUE previous ``n_lags``
    observations (a standard one-step nowcast), and coefficients are fit on train
    only. Returns (y_true, y_pred) over the test block.
    """
    y_train = train_df[_TARGET].to_numpy().astype(float)
    Xtr, ytr = build_ar_design(y_train, n_lags)
    if len(Xtr) == 0:                          # degenerate (very short train)
        return persistence_forecast(train_df, test_df)
    w, b = fit_linear_ar(Xtr, ytr)

    # Build test lags from the true series (train tail + test), so each test target
    # is predicted from its real past values -- no autoregressive error compounding.
    y_test = test_df[_TARGET].to_numpy().astype(float)
    full = np.concatenate([y_train[-n_lags:], y_test])
    Xte, _ = build_ar_design(full, n_lags)
    # The first len(y_test) design rows map to the test targets.
    Xte = Xte[: len(y_test)]
    y_pred = np.clip(Xte @ w + b, 0.0, None)
    return y_test, y_pred


# --------------------------------------------------------------------------- #
# 3) ARIMA (statsmodels) -- the canonical Box-Jenkins univariate forecaster
# --------------------------------------------------------------------------- #
def arima_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   order: tuple[int, int, int] = (2, 0, 1),
                   max_train: int = 4000) -> tuple[np.ndarray, np.ndarray] | None:
    """One-step-ahead ARIMA(p,d,q) forecast over the test block.

    Fits on train (capped at ``max_train`` most-recent points for speed), then uses
    statsmodels' ``append`` to roll the true test observations in one at a time and
    take the one-step forecast at each step (leak-free walk-forward). Returns
    (y_true, y_pred), or ``None`` if statsmodels is missing or the fit fails (the
    runner then simply omits ARIMA).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: WPS433
    except Exception:
        return None

    y_train = train_df[_TARGET].to_numpy().astype(float)
    y_test = test_df[_TARGET].to_numpy().astype(float)
    if len(y_train) > max_train:
        y_train = y_train[-max_train:]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(y_train, order=order).fit()
            preds = np.empty(len(y_test))
            for i in range(len(y_test)):
                preds[i] = float(np.asarray(res.forecast(steps=1))[0])
                # Roll the TRUE observation in for the next one-step forecast.
                res = res.append([y_test[i]], refit=False)
        return y_test, np.clip(preds, 0.0, None)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Orchestration entry point
# --------------------------------------------------------------------------- #
def run_before(train_df: pd.DataFrame, test_df: pd.DataFrame, n_lags: int = 6,
               include_arima: bool = True,
               arima_order: tuple[int, int, int] = (2, 0, 1)) -> dict:
    """Run the BEFORE baselines and return a metrics dict.

    Output schema (per method): {"rmse","mae","y_true":[...],"y_pred":[...]}.
    ``persistence`` and ``linear_ar`` are always present; ``arima`` is included when
    available. All forecasts are one-step-ahead and leak-free.
    """
    out: dict = {}

    def _record(name, y_true, y_pred):
        out[name] = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "y_true": np.asarray(y_true, float).tolist(),
            "y_pred": np.asarray(y_pred, float).tolist(),
        }

    yt, yp = persistence_forecast(train_df, test_df)
    _record("persistence", yt, yp)

    yt, yp = linear_ar_forecast(train_df, test_df, n_lags=n_lags)
    _record("linear_ar", yt, yp)

    if include_arima:
        arima = arima_forecast(train_df, test_df, order=arima_order)
        if arima is not None:
            _record("arima", arima[0], arima[1])

    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_airquality import synthetic_pm25
    from common.synthetic_climate import time_split

    df = synthetic_pm25(n_days=90, seed=0, freq="h")
    tr, va, te = time_split(df, 0.7, 0.15)
    res = run_before(pd.concat([tr, va]), te)
    for name, m in res.items():
        print(f"{name:12s} rmse={m['rmse']:.3f} mae={m['mae']:.3f}")
