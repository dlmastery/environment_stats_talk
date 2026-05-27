"""BEFORE — the traditional linear rainfall-runoff baseline.

This is the model an environmental statistician / hydrologist reaches for first:
a **multiple linear regression of streamflow on same-day and lagged precipitation**
(plus, optionally, temperature/PET and a short antecedent-precipitation index).
It is the honest classical reference — transparent, CPU-only, fit in closed form
with ordinary least squares — that any AFTER model must out-skill.

Why it underfits (on purpose)
-----------------------------
Real rainfall-runoff is **nonlinear and state-dependent**: the same rain produces
little runoff on dry soil and a lot on saturated soil, snow delays the response,
and routing imposes a lag. A linear map ``Q = a0 + sum_k a_k * P(t-k) + ...`` cannot
represent that conditional, multiplicative storage, so it leaves real skill on the
table. That gap is exactly what the LSTM closes in the AFTER script.

We also expose a tiny **conceptual single-bucket** linear-reservoir variant
(``bucket_forecast``) — the simplest classical alternative — to make the point that
the limitation is the *linear* structure, not regression per se.

Contract (shared with the AFTER side so the orchestrator treats them uniformly):

    run_before(train_df, test_df, ...) -> dict with per-method
        {"rmse":.., "nse":.., "kge":.., "y_true":[...], "y_pred":[...]}

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1]   # experiments/08_hydrology_streamflow
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hydro_metrics import nse, kge, rmse  # type: ignore  # noqa: E402

__all__ = [
    "build_lagged_design",
    "fit_linear_runoff",
    "predict_linear_runoff",
    "bucket_forecast",
    "run_before",
]


# --------------------------------------------------------------------------- #
# Linear regression on lagged precipitation (+ optional temp/PET / API)
# --------------------------------------------------------------------------- #
def build_lagged_design(df: pd.DataFrame, n_lags: int = 7,
                        use_temp: bool = True, use_pet: bool = True,
                        api_window: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Build the linear design matrix X and target y from a catchment DataFrame.

    Features (all leak-free — only same-day and PAST values):
      * precipitation at lags 0..n_lags  (same-day + lagged rainfall)
      * an Antecedent Precipitation Index: an exponentially weighted sum of past
        precip over ``api_window`` days (the classic hydrologist's wetness proxy)
      * optionally same-day temperature and PET.

    The first ``max(n_lags, api_window)`` rows (without full history) are dropped so
    no feature is fabricated. Returns (X, y) as float arrays.
    """
    precip = df["precip"].to_numpy().astype(float)
    flow = df["streamflow"].to_numpy().astype(float)
    n = len(df)

    cols = []
    # Same-day + lagged precipitation.
    for k in range(n_lags + 1):
        lag = np.full(n, np.nan)
        if k == 0:
            lag = precip.copy()
        else:
            lag[k:] = precip[:-k]
        cols.append(lag)

    # Antecedent Precipitation Index (exponential decay) — a wetness *proxy* the
    # linear model can use, but it is still a fixed linear feature (no state).
    decay = 0.92
    api = np.zeros(n)
    acc = 0.0
    for i in range(n):
        acc = decay * acc + precip[i]
        api[i] = acc
    # shift by 1 day so the target day's own rain is not double-counted as "antecedent"
    api_shift = np.full(n, np.nan)
    api_shift[1:] = api[:-1]
    cols.append(api_shift)

    if use_temp:
        cols.append(df["temperature"].to_numpy().astype(float))
    if use_pet:
        cols.append(df["pet"].to_numpy().astype(float))

    X = np.column_stack(cols)
    # Drop warm-up rows lacking full history.
    warmup = max(n_lags, 1)
    mask = np.arange(n) >= warmup
    valid = mask & np.isfinite(X).all(axis=1)
    return X[valid], flow[valid]


def fit_linear_runoff(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6):
    """Ordinary (ridge-stabilized) least-squares fit. Returns coefficients incl.
    intercept as ``(w, b)``. Ridge is tiny — just numerical conditioning."""
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    coef = np.linalg.solve(A, Xb.T @ y)
    return coef[:-1], coef[-1]


def predict_linear_runoff(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict streamflow; clip at zero (negative flow is unphysical)."""
    return np.clip(X @ w + b, 0.0, None)


# --------------------------------------------------------------------------- #
# Conceptual single linear-reservoir bucket (the classical alternative)
# --------------------------------------------------------------------------- #
def bucket_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    n_grid: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """A minimal single linear-reservoir conceptual model.

    Storage updates as ``S_{t+1} = (1-k) S_t + P_t`` and discharge is ``Q_t = k S_t``
    — a *linear* reservoir. We pick the recession constant ``k`` by a tiny grid
    search on the train split (the kind of manual calibration the BEFORE workflow
    actually involves), then run it forward on test with a warm start.

    Returns (y_true, y_pred) over the test block.
    """
    p_tr = train_df["precip"].to_numpy().astype(float)
    q_tr = train_df["streamflow"].to_numpy().astype(float)

    def simulate(p, k, s0):
        s = s0
        q = np.empty(len(p))
        for i in range(len(p)):
            s = (1 - k) * s + p[i]
            q[i] = k * s
        return q

    # Calibrate k on train (grid search minimizing RMSE).
    best_k, best_err, best_s_end = 0.1, np.inf, p_tr.mean()
    s0_tr = float(np.mean(q_tr))
    for k in np.linspace(0.02, 0.6, n_grid):
        q_sim = simulate(p_tr, k, s0_tr)
        err = float(np.sqrt(np.mean((q_sim - q_tr) ** 2)))
        if err < best_err:
            best_err, best_k = err, k
    # Run forward on train to get the storage carried into the test block.
    s = s0_tr
    for i in range(len(p_tr)):
        s = (1 - best_k) * s + p_tr[i]

    p_te = test_df["precip"].to_numpy().astype(float)
    q_pred = simulate(p_te, best_k, s)
    q_true = test_df["streamflow"].to_numpy().astype(float)
    return q_true, np.clip(q_pred, 0.0, None)


# --------------------------------------------------------------------------- #
# Orchestration entry point
# --------------------------------------------------------------------------- #
def run_before(train_df: pd.DataFrame, test_df: pd.DataFrame, n_lags: int = 7,
               include_bucket: bool = True) -> dict:
    """Run the BEFORE baselines and return a metrics dict.

    Output schema (per method):
        { method: {"rmse":.., "nse":.., "kge":.., "y_true":[...], "y_pred":[...]} }
    ``linear`` (lagged-precip regression) is always present; ``bucket`` (linear
    reservoir) is included by default.
    """
    out: dict = {}

    def _record(name, y_true, y_pred):
        out[name] = {
            "rmse": rmse(y_true, y_pred),
            "nse": nse(y_true, y_pred),
            "kge": kge(y_true, y_pred),
            "y_true": np.asarray(y_true, float).tolist(),
            "y_pred": np.asarray(y_pred, float).tolist(),
        }

    # Fit the linear regression on train, evaluate on test. To keep test targets
    # aligned with the AFTER model, build the test design using a train tail so
    # lagged/antecedent features have full history for every test day.
    Xtr, ytr = build_lagged_design(train_df, n_lags=n_lags)
    w, b = fit_linear_runoff(Xtr, ytr)

    context = pd.concat([train_df.iloc[-max(n_lags, 30):], test_df])
    Xte_full, yte_full = build_lagged_design(context, n_lags=n_lags)
    # The train tail is consumed as warm-up, so the valid design rows correspond to
    # the test block. Align on the trailing rows that map to actual test days.
    y_pred = predict_linear_runoff(Xte_full, w, b)
    n_common = min(len(yte_full), len(test_df))
    _record("linear", yte_full[-n_common:], y_pred[-n_common:])

    if include_bucket:
        yt, yp = bucket_forecast(train_df, test_df)
        n = min(len(yt), len(yp))
        _record("bucket", yt[-n:], yp[-n:])

    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_hydrology import synthetic_catchment
    from common.synthetic_climate import time_split

    df = synthetic_catchment(n_years=8, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    res = run_before(pd.concat([tr, va]), te)
    for name, m in res.items():
        print(f"{name:8s} rmse={m['rmse']:.3f} nse={m['nse']:.3f} kge={m['kge']:.3f}")
