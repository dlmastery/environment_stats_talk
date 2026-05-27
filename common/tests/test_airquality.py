"""FAST unit tests for the synthetic PM2.5 / air-quality generator.

Run from the repo root:  python -m pytest common/tests/test_airquality.py -q

These guarantee the synthetic series is deterministic, well-shaped, physically sane
(non-negative PM2.5, plausible columns), and -- the crux of the experiment -- that
PM2.5 genuinely RESPONDS to the weather covariates: wind ventilates (negative
correlation), a collapsing boundary layer / inversion traps pollution (PM2.5 rises
as boundary-layer height falls), and stagnation episodes produce spikes. If these
hold, a covariate model can beat a history-only baseline.
"""
from __future__ import annotations

import numpy as np

# Import the generator from the submodule directly (intentionally NOT in
# common/__init__.py).
from common.synthetic_airquality import (
    synthetic_pm25, AirQualityParams, exceedance_threshold,
)

_COLS = ["pm25", "wind", "temp", "boundary_layer", "hour", "dow", "is_weekend"]


def test_schema_and_determinism():
    a = synthetic_pm25(n_days=20, seed=7, freq="h")
    b = synthetic_pm25(n_days=20, seed=7, freq="h")
    assert a.shape == (20 * 24, len(_COLS))
    assert list(a.columns) == _COLS
    assert a.index.name == "date"
    assert np.allclose(a["pm25"].to_numpy(), b["pm25"].to_numpy())


def test_daily_frequency_shape():
    d = synthetic_pm25(n_days=30, seed=0, freq="D")
    assert d.shape == (30, len(_COLS))
    # Daily series has no intra-day hour signal (hour column is 0).
    assert (d["hour"].to_numpy() == 0).all()


def test_pm25_nonnegative_and_finite():
    df = synthetic_pm25(n_days=40, seed=1, freq="h")
    pm = df["pm25"].to_numpy()
    assert (pm >= 0).all()
    assert np.isfinite(df.to_numpy()).all()
    assert pm.max() > 0  # there is pollution
    assert (df["wind"].to_numpy() > 0).all()
    assert (df["boundary_layer"].to_numpy() > 0).all()


def test_pm25_responds_to_wind_ventilation():
    """Wind disperses pollution: PM2.5 should correlate NEGATIVELY with wind."""
    df = synthetic_pm25(n_days=120, seed=0, freq="h")
    r = float(np.corrcoef(df["pm25"], df["wind"])[0, 1])
    assert r < -0.05   # ventilation: more wind -> cleaner air


def test_pm25_responds_to_inversion_boundary_layer():
    """A shallow (low) boundary layer / inversion traps pollution: PM2.5 should
    correlate NEGATIVELY with boundary-layer height (shallow layer -> high PM2.5)."""
    df = synthetic_pm25(n_days=120, seed=0, freq="h")
    r = float(np.corrcoef(df["pm25"], df["boundary_layer"])[0, 1])
    assert r < -0.2    # the dominant dilution driver


def test_stronger_inversion_raises_pollution_extremes():
    """A larger inversion_strength should trap more during shallow-layer periods,
    raising the upper tail of PM2.5 (sharper episodes)."""
    weak = synthetic_pm25(n_days=90, seed=3, freq="h",
                          params=AirQualityParams(inversion_strength=1.0))
    strong = synthetic_pm25(n_days=90, seed=3, freq="h",
                            params=AirQualityParams(inversion_strength=2.6))
    q_weak = float(np.quantile(weak["pm25"], 0.99))
    q_strong = float(np.quantile(strong["pm25"], 0.99))
    assert q_strong > q_weak


def test_exceedance_threshold_positive_rate():
    """The 90th-percentile threshold should label ~10% of points as exceedances."""
    df = synthetic_pm25(n_days=80, seed=2, freq="h")
    pm = df["pm25"].to_numpy()
    thr = exceedance_threshold(pm, 0.90)
    frac = float(np.mean(pm > thr))
    assert 0.05 < frac < 0.15
    assert thr > 0
