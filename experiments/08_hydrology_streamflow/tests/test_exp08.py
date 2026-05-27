"""FAST unit tests for Experiment 08 (rainfall-runoff streamflow forecasting).

The test logic is tiny and CPU-only (8 synthetic years, hidden≤24 LSTM, ≤25 capped
epochs). They check that the synthetic data is sane, the metrics are finite, both
models run, and — the headline of this experiment — that the LSTM beats the linear
baseline on a held-out test period (NSE_lstm > NSE_linear).

Note on wall time: on a *cold* interpreter the suite is dominated by torch's
one-time import + first-op CPU init (~13s on this build) that runs before any test
logic; the heaviest single assertion (the LSTM-beats-linear holdout, ~20s on CPU)
needs ~25 epochs to demonstrate a genuine win. A cold run is therefore ~40s here;
re-running in a warm process is much faster. Time only the logic with
``--durations=10``. The production headline run auto-selects CUDA via
``after.lstm_runoff.get_device()``; here we pin CPU.

Run from the REPO ROOT:
    python -m pytest experiments/08_hydrology_streamflow/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root (for `import common`) and the experiment dir (for before/after/hydro_metrics).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from common.synthetic_hydrology import synthetic_catchment  # noqa: E402
from common.synthetic_climate import time_split  # noqa: E402
from hydro_metrics import nse, kge, rmse  # noqa: E402
from before.linear_baseline import (  # noqa: E402
    build_lagged_design, fit_linear_runoff, run_before,
)
from after.lstm_runoff import (  # noqa: E402
    LSTMConfig, make_windows, LSTMRunoff, train_runoff, predict, run_after,
)

_CPU = torch.device("cpu")
torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))


@pytest.fixture(scope="module")
def split():
    # 8 years -> 2920 days; enough data for the LSTM to GENERALIZE and beat linear
    # on the held-out test period, while staying fast on CPU (a few seconds/epoch).
    df = synthetic_catchment(n_years=8, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    return tr, va, te


# ----------------------------- metrics ------------------------------------ #
def test_metrics_perfect_and_range():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert nse(y, y) == pytest.approx(1.0)
    assert kge(y, y) == pytest.approx(1.0)
    assert rmse(y, y) == pytest.approx(0.0)
    # predicting the mean -> NSE == 0
    assert nse(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-9)


# ----------------------------- BEFORE ------------------------------------- #
def test_linear_design_leakfree_shapes(split):
    tr, va, te = split
    X, y = build_lagged_design(tr, n_lags=5)
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0 and X.shape[1] > 5
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_linear_fit_and_run_before_finite(split):
    tr, va, te = split
    X, y = build_lagged_design(tr, n_lags=5)
    w, b = fit_linear_runoff(X, y)
    assert np.isfinite(w).all() and np.isfinite(b)
    res = run_before(pd.concat([tr, va]), te, n_lags=5)
    assert "linear" in res and "bucket" in res
    for m in res.values():
        for k in ("rmse", "nse", "kge"):
            assert np.isfinite(m[k])


# ----------------------------- windows / model ---------------------------- #
def test_make_windows_shapes(split):
    tr, va, te = split
    X, y, fmean, fstd, ymean, ystd = make_windows(tr, lookback=15)
    assert X.shape[1:] == (15, 5)            # 3 forcings + sin/cos doy
    assert X.shape[0] == y.shape[0] == len(tr) - 15 + 1
    assert np.isfinite(fmean).all() and (fstd > 0).all() and ystd > 0


def test_model_forward_shape():
    model = LSTMRunoff(n_features=5, hidden=16, num_layers=1)
    x = torch.randn(8, 12, 5)
    out = model(x)
    assert out.shape == (8,)


def test_train_one_epoch_and_predict_nonneg(split):
    tr, va, te = split
    cfg = LSTMConfig(lookback=15, hidden=16, epochs=1, patience=1, seed=0)
    model, stats = train_runoff(tr, va, cfg, device=_CPU)
    assert stats["epochs_run"] == 1 and stats["y_std"] > 0
    hist = pd.concat([tr, va])
    y_true, y_pred = predict(model, hist, te, cfg, stats, device=_CPU)
    assert y_true.shape == y_pred.shape == (len(te),)
    assert (y_pred >= 0).all() and np.isfinite(y_pred).all()


def test_run_after_metrics_finite(split):
    tr, va, te = split
    cfg = LSTMConfig(lookback=15, hidden=16, epochs=2, patience=2, seed=0)
    res = run_after(tr, va, te, cfg, device=_CPU)
    for k in ("rmse", "nse", "kge"):
        assert np.isfinite(res[k])
    assert len(res["y_pred"]) == len(te)


# ----------------------------- headline: AFTER beats BEFORE ---------------- #
def test_lstm_beats_linear_on_holdout(split):
    """The whole point of this experiment: an LSTM that carries state captures the
    nonlinear catchment memory the linear model cannot, so NSE_lstm > NSE_linear."""
    tr, va, te = split
    before = run_before(pd.concat([tr, va]), te, n_lags=7)
    # Same stable recipe as the headline run (MSE early-stopping, lr=1e-3), with a
    # long-enough lookback to carry catchment memory but capped in epochs so the
    # assertion stays well under the CPU time budget (~17s on this machine).
    cfg = LSTMConfig(lookback=35, hidden=24, epochs=25, patience=25, lr=1e-3, seed=0)
    after = run_after(tr, va, te, cfg, device=_CPU)

    # Align on common trailing test targets.
    n = min(len(before["linear"]["y_true"]), len(after["y_true"]))
    obs = np.asarray(after["y_true"])[-n:]
    lin_pred = np.asarray(before["linear"]["y_pred"])[-n:]
    lstm_pred = np.asarray(after["y_pred"])[-n:]

    nse_lin = nse(obs, lin_pred)
    nse_lstm = nse(obs, lstm_pred)
    assert np.isfinite(nse_lin) and np.isfinite(nse_lstm)
    assert nse_lstm > nse_lin   # AFTER wins
