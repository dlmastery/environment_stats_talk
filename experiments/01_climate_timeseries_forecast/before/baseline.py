"""BEFORE — traditional time-series baselines for daily 2m-temperature forecasting.

These are the methods an environmental statistician reaches for *before* any deep
learning: cheap, transparent, CPU-only, and surprisingly hard to beat at short
horizons. They are the honest reference that any AFTER model must out-skill.

Methods implemented
--------------------
- ``persistence_forecast``      : the last observed value carried forward h steps
                                  (a.k.a. random walk / "naive"). The canonical
                                  reference for forecast skill scores.
- ``seasonal_naive_forecast``   : the value from one seasonal period (365 days)
                                  ago — captures the annual cycle for free.
- ``sarima_forecast``           : a classical autoregressive model fit with
                                  statsmodels and rolled forward over the test set.
                                  Prefers SARIMAX when its statespace extensions
                                  import cleanly, otherwise uses statsmodels'
                                  ``AutoReg`` (an AR(p) on the de-trended signal) —
                                  both are interpretable, CPU-only classical
                                  baselines. The function name is kept for API
                                  stability across the before/after orchestration.

All forecasters share one contract so the orchestrator can treat them uniformly:

    forecast(train_series, test_series, h) -> (y_true, y_pred)

where ``y_true``/``y_pred`` are 1-D numpy arrays aligned on the test targets that
can actually be formed at horizon ``h`` (so RMSE/MAE/ACC are computed on the same
targets for every method, BEFORE and AFTER).

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make `import common` work whether invoked via pytest (rootdir on path) or as a
# script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics  # noqa: E402

__all__ = [
    "persistence_forecast",
    "seasonal_naive_forecast",
    "sarima_forecast",
    "run_before",
    "SEASONAL_PERIOD",
]

SEASONAL_PERIOD = 365  # daily data, annual cycle


def _as_series(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy().ravel()
    return np.asarray(x, dtype=float).ravel()


def persistence_forecast(train, test, h: int = 1):
    """Persistence (random-walk) forecast: y_hat(t+h) = y(t).

    On the *test* block we predict each target ``test[i]`` (for i >= h) using the
    value ``h`` steps earlier. The very first ``h`` targets use the tail of train,
    so no target is wasted and there is no leakage.

    Returns (y_true, y_pred) as aligned 1-D arrays of length ``len(test)``.
    """
    tr, te = _as_series(train), _as_series(test)
    full = np.concatenate([tr, te])
    n_tr = len(tr)
    # Targets are the whole test block; predictor is the value h steps before each.
    y_true = te
    y_pred = full[n_tr - h: n_tr - h + len(te)]
    return y_true, y_pred


def seasonal_naive_forecast(train, test, h: int = 1, period: int = SEASONAL_PERIOD):
    """Seasonal-naive forecast: y_hat(t+h) = y(t + h - period).

    Uses the observation from one seasonal period (``period`` days) before the
    target. Falls back to persistence when not enough history precedes a target
    (only possible for very short training series).

    Returns (y_true, y_pred) aligned 1-D arrays of length ``len(test)``.
    """
    tr, te = _as_series(train), _as_series(test)
    full = np.concatenate([tr, te])
    n_tr = len(tr)
    y_true = te
    y_pred = np.empty(len(te), dtype=float)
    for i in range(len(te)):
        target_idx = n_tr + i              # absolute index of the target in `full`
        src = target_idx - period          # one season before the target
        if src >= 0:
            y_pred[i] = full[src]
        else:                              # not enough history -> persistence
            y_pred[i] = full[max(target_idx - h, 0)]
    return y_true, y_pred


def _sarimax_available() -> bool:
    """True iff statsmodels' statespace (SARIMAX) extensions import cleanly.

    In some environments the compiled statespace modules have a numpy ABI
    mismatch; we detect that here and fall back to the pure-AutoReg path so the
    classical baseline always runs.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
        return True
    except Exception:
        return False


def _autoreg_rolling(tr: np.ndarray, te: np.ndarray, h: int, lags: int,
                     max_train: int) -> np.ndarray:
    """Rolling h-step AutoReg forecast using fixed fitted coefficients.

    Fit an AR(``lags``) once on the (de-meaned) train tail with statsmodels'
    ``AutoReg`` (the classical, non-statespace estimator), then roll forward over
    the test block. At each step we recursively forecast ``h`` steps using the most
    recent *observed* values as context (no leakage), then advance by feeding the
    realized observation in. This is O(len(test) * h * lags) — fast and CPU-only.
    """
    from statsmodels.tsa.ar_model import AutoReg

    tr = tr[-max_train:] if len(tr) > max_train else tr
    mu = float(np.mean(tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = AutoReg(tr - mu, lags=lags, old_names=False).fit()
    params = np.asarray(res.params, dtype=float)
    const = params[0]
    ar = params[1:1 + lags]  # AR coefficients phi_1..phi_p

    full = np.concatenate([tr, te]) - mu  # de-meaned full series
    n_tr = len(tr)
    y_pred = np.empty(len(te), dtype=float)
    for i in range(len(te)):
        # Context = observed values up to (but not including) the target's window.
        hist = list(full[: n_tr + i][-lags:])
        if len(hist) < lags:                       # pad short history with zeros (mean)
            hist = [0.0] * (lags - len(hist)) + hist
        for _ in range(h):                         # recursive multi-step forecast
            nxt = const + float(np.dot(ar, hist[-lags:][::-1]))
            hist.append(nxt)
        y_pred[i] = hist[-1] + mu
    return y_pred


def sarima_forecast(train, test, h: int = 1, order=(2, 0, 1),
                    seasonal_order=(0, 0, 0, 0), max_train: int = 2000):
    """Classical autoregressive forecast rolled forward over the test set.

    Prefers statsmodels ``SARIMAX`` (fit once on the recent train tail, then rolled
    forward one observation at a time — a true out-of-sample rolling forecast, no
    leakage). If the statespace extensions are unavailable in this environment
    (numpy ABI mismatch), transparently falls back to a rolling ``AutoReg`` AR(p)
    forecast of equivalent order. We cap the fit length with ``max_train`` so the
    classical baseline stays fast on CPU.

    The default ``order=(2,0,1)`` with no seasonal term is a deliberately light
    ARMA on the de-meaned weather signal — the seasonal cycle is better handled by
    ``seasonal_naive``; including a 365-day seasonal AR is prohibitively slow,
    which is itself part of the BEFORE story.

    Returns (y_true, y_pred) aligned 1-D arrays of length ``len(test)``.
    """
    tr, te = _as_series(train), _as_series(test)

    if not _sarimax_available():
        lags = max(int(order[0]), 1)
        y_pred = _autoreg_rolling(tr, te, h=h, lags=lags, max_train=max_train)
        return te, y_pred

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    tr_fit = tr[-max_train:] if len(tr) > max_train else tr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter is expected & benign
        model = SARIMAX(
            tr_fit, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        y_pred = np.empty(len(te), dtype=float)
        for i in range(len(te)):
            fc = res.forecast(steps=h)
            y_pred[i] = float(np.asarray(fc).ravel()[-1])
            # Feed the realized observation forward (rolling 1-update of state).
            res = res.append([te[i]], refit=False)
    return te, y_pred


def run_before(train, test, h: int = 1, include_sarima: bool = True,
               sarima_order=(2, 0, 1)) -> dict:
    """Run all BEFORE baselines at horizon ``h`` and return a metrics dict.

    Output schema (per method):
        { method: {"rmse":.., "mae":.., "acc":.., "y_true":[...], "y_pred":[...]} }
    ``persistence`` is always present (it is the skill-score reference).
    """
    out: dict = {}

    def _record(name, y_true, y_pred):
        out[name] = {
            "rmse": metrics.rmse(y_true, y_pred),
            "mae": metrics.mae(y_true, y_pred),
            "acc": metrics.anomaly_correlation(y_true, y_pred),
            "y_true": np.asarray(y_true, float).tolist(),
            "y_pred": np.asarray(y_pred, float).tolist(),
        }

    yt, yp = persistence_forecast(train, test, h)
    _record("persistence", yt, yp)

    yt, yp = seasonal_naive_forecast(train, test, h)
    _record("seasonal_naive", yt, yp)

    if include_sarima:
        yt, yp = sarima_forecast(train, test, h, order=sarima_order)
        _record("sarima", yt, yp)

    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import daily_temperature, time_split

    df = daily_temperature(n_years=6, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    res = run_before(tr["t2m"], te["t2m"], h=1, include_sarima=True)
    for name, m in res.items():
        print(f"{name:16s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} acc={m['acc']:.3f}")
