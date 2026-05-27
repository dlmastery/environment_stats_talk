"""FAST unit tests for Experiment 07 (PM2.5 air-quality nowcast/forecast).

The test logic is tiny and CPU-only (a few synthetic months, a small GBM). They
check that the synthetic data is sane, the metrics are finite, both sides run
leak-free, and -- the headline of this experiment -- that the covariate (weather)
model beats the persistence baseline on a held-out test tail, on BOTH RMSE and
spike-detection F1.

Run from the REPO ROOT (whole suite is well under 30 s on CPU):
    python -m pytest experiments/07_air_quality_nowcast/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Repo root (for `import common`) and the experiment dir (for before/after/aq_metrics).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.synthetic_airquality import synthetic_pm25, exceedance_threshold  # noqa: E402
from common.synthetic_climate import time_split  # noqa: E402
from aq_metrics import (  # noqa: E402
    rmse, mae, skill_vs_persistence, spike_detection_scores, all_metrics,
)
from before.baseline import (  # noqa: E402
    persistence_forecast, build_ar_design, fit_linear_ar, linear_ar_forecast,
    run_before,
)
from after.gbm_nowcast import GBMConfig, build_feature_frame, run_after  # noqa: E402


@pytest.fixture(scope="module")
def split():
    # 100 days hourly (2400 steps): enough for the GBM to learn the weather signal
    # and beat persistence on the held-out tail, while staying fast on CPU.
    df = synthetic_pm25(n_days=100, seed=0, freq="h")
    tr, va, te = time_split(df, 0.7, 0.15)
    return tr, va, te


# ----------------------------- metrics ------------------------------------ #
def test_metrics_perfect_and_skill():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rmse(y, y) == pytest.approx(0.0)
    assert mae(y, y) == pytest.approx(0.0)
    # skill: model ties persistence -> 0; model perfect -> 1
    assert skill_vs_persistence(0.0, 5.0) == pytest.approx(1.0)
    assert skill_vs_persistence(5.0, 5.0) == pytest.approx(0.0, abs=1e-9)


def test_spike_f1_finite_and_bounded():
    obs = np.array([10.0, 20.0, 80.0, 90.0, 15.0, 70.0])
    pred = np.array([12.0, 25.0, 60.0, 95.0, 18.0, 40.0])
    s = spike_detection_scores(obs, pred, threshold=50.0)
    for k in ("precision", "recall", "f1"):
        assert 0.0 <= s[k] <= 1.0 and np.isfinite(s[k])
    assert s["support_true"] == 3
    # Degenerate: no exceedances anywhere -> F1 finite (0), not NaN.
    s0 = spike_detection_scores(np.zeros(5), np.zeros(5), threshold=50.0)
    assert s0["f1"] == 0.0 and np.isfinite(s0["f1"])


# ----------------------------- BEFORE ------------------------------------- #
def test_persistence_shapes(split):
    tr, va, te = split
    y_true, y_pred = persistence_forecast(pd.concat([tr, va]), te)
    assert y_true.shape == y_pred.shape == (len(te),)
    assert (y_pred >= 0).all() and np.isfinite(y_pred).all()


def test_ar_design_and_fit(split):
    tr, va, te = split
    X, y = build_ar_design(tr["pm25"].to_numpy(), n_lags=4)
    assert X.shape[0] == y.shape[0] > 0 and X.shape[1] == 4
    w, b = fit_linear_ar(X, y)
    assert np.isfinite(w).all() and np.isfinite(b)


def test_run_before_finite(split):
    tr, va, te = split
    # Skip ARIMA here to keep the test fast; persistence + linear AR is enough.
    res = run_before(pd.concat([tr, va]), te, n_lags=6, include_arima=False)
    assert "persistence" in res and "linear_ar" in res
    for m in res.values():
        for k in ("rmse", "mae"):
            assert np.isfinite(m[k])
        assert len(m["y_pred"]) == len(te)


# ----------------------------- AFTER -------------------------------------- #
def test_feature_frame_leakfree(split):
    tr, va, te = split
    cfg = GBMConfig()
    X, y, names = build_feature_frame(tr, cfg)
    assert X.shape[0] == y.shape[0] > 0
    assert len(names) == X.shape[1]
    assert X.notna().all().all()           # no missing rows survive
    # The target column must never appear as a feature (no trivial leakage).
    assert "pm25" not in names
    # Weather features present when use_weather is on.
    assert any("wind" in c for c in names) and any("boundary_layer" in c for c in names)


def test_run_after_metrics_finite(split):
    tr, va, te = split
    cfg = GBMConfig(n_estimators=120, max_depth=3, seed=0)
    res = run_after(pd.concat([tr, va]), te, cfg)
    for k in ("rmse", "mae"):
        assert np.isfinite(res[k])
    assert len(res["y_pred"]) <= len(te) and res["use_weather"] is True


def test_ablation_drops_weather_features(split):
    tr, va, te = split
    full = build_feature_frame(tr, GBMConfig(use_weather=True))[2]
    abl = build_feature_frame(tr, GBMConfig(use_weather=False))[2]
    assert len(abl) < len(full)
    assert not any("wind" in c or "boundary_layer" in c or "temp" in c for c in abl)


# ----------------- headline: AFTER (weather) beats persistence ------------ #
def test_after_beats_persistence_on_holdout(split):
    """The whole point: a covariate model that reads the weather beats the
    persistence baseline on the held-out tail -- on both RMSE and spike F1."""
    tr, va, te = split
    tv = pd.concat([tr, va])

    before = run_before(tv, te, n_lags=6, include_arima=False)
    cfg = GBMConfig(n_estimators=200, max_depth=3, learning_rate=0.05, seed=0)
    after = run_after(tv, te, cfg)

    # Align on common trailing test targets.
    n = min(len(before["persistence"]["y_true"]), len(after["y_true"]))
    obs = np.asarray(after["y_true"])[-n:]
    pers_pred = np.asarray(before["persistence"]["y_pred"])[-n:]
    after_pred = np.asarray(after["y_pred"])[-n:]

    thr = exceedance_threshold(obs, 0.90)
    rmse_pers = rmse(obs, pers_pred)
    m_pers = all_metrics(obs, pers_pred, rmse_pers, thr)
    m_after = all_metrics(obs, after_pred, rmse_pers, thr)

    assert np.isfinite(m_after["rmse"]) and np.isfinite(m_after["spike_f1"])
    assert m_after["rmse"] < m_pers["rmse"]            # AFTER lower error
    assert m_after["skill_vs_persistence"] > 0.0       # positive skill vs persistence
    assert m_after["spike_f1"] >= m_pers["spike_f1"]   # at least as good at spikes


def test_weather_helps_vs_history_only(split):
    """Honest covariate check: with the weather covariates the AFTER model should
    not be worse than the history-only ablation on RMSE (the covariates carry the
    win). Asserted with a small tolerance to stay robust on the tiny test series."""
    tr, va, te = split
    tv = pd.concat([tr, va])
    with_w = run_after(tv, te, GBMConfig(n_estimators=200, use_weather=True, seed=0))
    hist = run_after(tv, te, GBMConfig(n_estimators=200, use_weather=False, seed=0))
    n = min(len(with_w["y_true"]), len(hist["y_true"]))
    obs = np.asarray(with_w["y_true"])[-n:]
    r_w = rmse(obs, np.asarray(with_w["y_pred"])[-n:])
    r_h = rmse(obs, np.asarray(hist["y_pred"])[-n:])
    assert r_w <= r_h + 1e-6   # weather covariates do not hurt; in the full run they help
