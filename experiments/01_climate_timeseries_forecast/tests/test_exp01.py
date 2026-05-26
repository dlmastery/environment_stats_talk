"""FAST unit tests for Experiment 01 (climate time-series forecasting).

The *test logic* is tiny and CPU-only: 2-year synthetic series, hidden=8 LSTM,
1 epoch, a 10-point rolling AR fit. They check shapes, that the baselines run,
that the LSTM trains one step and predicts the right shape, and that all metrics
are finite. Warm-interpreter execution of the actual assertions is only a few
seconds.

Note on wall time: on a *cold* interpreter the suite is dominated by one-time
library start-up that is outside the test logic — importing torch (~8s on this
conda build) and statsmodels (~4s), plus torch's first-op CPU backend init (~9s).
That puts a cold run around ~30s here. Re-running in the same process, or on a
machine with warmed import caches, is far faster. To time only the logic:
``python -m pytest ... --durations=10`` (per-call times are sub-second except the
two that absorb those one-time inits).

Run from the REPO ROOT:
    python -m pytest experiments/01_climate_timeseries_forecast/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root (for `import common`) and the experiment dir (for before/after pkgs).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from common import daily_temperature, time_split, metrics  # noqa: E402
from before.baseline import (  # noqa: E402
    persistence_forecast, seasonal_naive_forecast, sarima_forecast, run_before,
)
from after.dl_forecaster import (  # noqa: E402
    TrainConfig, make_windows, LSTMForecaster, train_forecaster, predict,
    run_after, zero_shot_foundation_baseline,
)

# Tests run on CPU (fast, deterministic, no GPU dependency). The production run
# auto-selects CUDA via after.dl_forecaster.get_device(); here we pin CPU and cap
# threads so the suite stays well under the 30s budget.
_CPU = torch.device("cpu")
torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))


# Small shared fixture: 2 years of daily data -> 730 points, splits cleanly and
# keeps every LSTM training step fast on CPU.
@pytest.fixture(scope="module")
def split():
    df = daily_temperature(n_years=2, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    return tr["t2m"], va["t2m"], te["t2m"]


# ----------------------------- BEFORE ------------------------------------- #
def test_persistence_shapes_and_alignment(split):
    tr, va, te = split
    y_true, y_pred = persistence_forecast(tr, te, h=1)
    assert y_true.shape == y_pred.shape == (len(te),)
    assert np.isfinite(y_true).all() and np.isfinite(y_pred).all()


def test_persistence_h_step(split):
    tr, va, te = split
    for h in (1, 7):
        y_true, y_pred = persistence_forecast(tr, te, h=h)
        assert len(y_true) == len(te) == len(y_pred)


def test_seasonal_naive_runs(split):
    tr, va, te = split
    y_true, y_pred = seasonal_naive_forecast(tr, te, h=1)
    assert y_true.shape == y_pred.shape == (len(te),)
    assert np.isfinite(y_pred).all()


def test_run_before_metrics_finite_no_sarima(split):
    tr, va, te = split
    res = run_before(tr, te, h=1, include_sarima=False)
    assert "persistence" in res and "seasonal_naive" in res
    for m in res.values():
        for k in ("rmse", "mae", "acc"):
            assert np.isfinite(m[k])


def test_sarima_runs_quickly(split):
    # Tiny train cap + few test points keeps this fast (the first statsmodels fit
    # also pays the library's one-time import cost); just confirm finite output.
    tr, va, te = split
    y_true, y_pred = sarima_forecast(tr, te.iloc[:10], h=1,
                                     order=(1, 0, 0), max_train=150)
    assert len(y_pred) == 10
    assert np.isfinite(y_pred).all()


# ----------------------------- windows ------------------------------------ #
def test_make_windows_shapes():
    s = daily_temperature(n_years=1, seed=1)["t2m"]
    X, y, mean, std = make_windows(s, lookback=10, horizon=1)
    assert X.shape[1:] == (10, 3)
    assert X.shape[0] == y.shape[0] == len(s) - 10 - 1 + 1
    assert np.isfinite(mean) and std > 0


def test_make_windows_uses_given_stats():
    s = daily_temperature(n_years=1, seed=2)["t2m"]
    X, y, mean, std = make_windows(s, lookback=5, horizon=7, mean=10.0, std=2.0)
    assert mean == 10.0 and std == 2.0
    assert X.shape[1:] == (5, 3)


# ----------------------------- AFTER (model) ------------------------------ #
def test_model_forward_shape():
    model = LSTMForecaster(n_features=3, hidden=8, num_layers=1)
    import torch
    x = torch.randn(4, 12, 3)
    out = model(x)
    assert out.shape == (4,)


def test_train_one_epoch_and_predict_shape(split):
    tr, va, te = split
    cfg = TrainConfig(lookback=12, horizon=1, hidden=8, epochs=1, patience=1, seed=0)
    model, stats = train_forecaster(tr, va, cfg, device=_CPU)
    assert stats["epochs_run"] == 1
    assert np.isfinite(stats["mean"]) and stats["std"] > 0
    hist = pd.concat([tr, va])
    y_true, y_pred = predict(model, hist, te, cfg, stats, device=_CPU)
    assert y_true.shape == y_pred.shape
    assert len(y_pred) == len(te)
    assert np.isfinite(y_pred).all()


def test_run_after_metrics_finite(split):
    tr, va, te = split
    cfg = TrainConfig(lookback=12, horizon=1, hidden=8, epochs=1, patience=1, seed=0)
    res = run_after(tr, va, te, cfg, device=_CPU)
    for k in ("rmse", "mae", "acc"):
        assert np.isfinite(res[k])
    assert res["epochs_run"] == 1
    assert len(res["y_pred"]) == len(te)


def test_run_after_horizon7_shape(split):
    tr, va, te = split
    cfg = TrainConfig(lookback=12, horizon=7, hidden=8, epochs=1, patience=1, seed=0)
    res = run_after(tr, va, te, cfg, device=_CPU)
    assert len(res["y_pred"]) == len(te)
    assert np.isfinite(res["rmse"])


# ----------------------------- foundation (optional) ---------------------- #
def test_foundation_baseline_optional(split):
    # Must not raise; returns None when Chronos/TimesFM is absent.
    tr, va, te = split
    out = zero_shot_foundation_baseline(pd.concat([tr, va]), te.iloc[:10], h=1)
    assert out is None or (len(out) == 2)


# ----------------------------- skill score -------------------------------- #
def test_skill_score_alignment(split):
    tr, va, te = split
    yp_t, yp_p = persistence_forecast(tr, te, h=1)
    cfg = TrainConfig(lookback=12, horizon=1, hidden=8, epochs=1, patience=1, seed=0)
    res = run_after(tr, va, te, cfg, device=_CPU)
    n = min(len(yp_p), len(res["y_pred"]))
    pers_rmse = metrics.rmse(np.asarray(res["y_true"])[-n:], yp_p[-n:])
    after_rmse = res["rmse"]
    sk = metrics.skill_score(after_rmse, pers_rmse, higher_is_better=False)
    assert np.isfinite(sk)
