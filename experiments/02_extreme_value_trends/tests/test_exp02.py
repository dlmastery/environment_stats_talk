"""Fast unit tests for experiment 02 (extreme-value & trend detection).

Designed to finish in well under 30s: short synthetic records and small bootstrap
counts. Run from the REPO ROOT:

    python -m pytest experiments/02_extreme_value_trends/tests -q

Coverage:
  * Mann-Kendall returns a finite slope & p-value and detects the injected trend
    sign (both increasing and decreasing series).
  * GEV fit returns finite parameters and monotonically increasing return levels.
  * Bootstrap CIs bracket the point estimate (lo <= point <= hi).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make the experiment dir importable so `before`/`after` packages resolve when
# pytest is launched from the repo root.
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from before.manual_eda import mann_kendall, empirical_return_level  # noqa: E402
from after.agentic_pipeline import (  # noqa: E402
    fit_gev, gev_return_level, bootstrap_return_levels, validate, lag1_autocorr,
)


# ------------------------------- Mann-Kendall ------------------------------- #
def test_mann_kendall_finite_outputs():
    rng = np.random.default_rng(0)
    y = rng.normal(size=30)
    res = mann_kendall(y)
    assert np.isfinite(res["sen_slope"])
    assert np.isfinite(res["p_value"])
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["n"] == 30


def test_mann_kendall_detects_increasing_trend():
    # Strong upward trend + small noise -> positive Sen slope, significant p.
    rng = np.random.default_rng(1)
    t = np.arange(40)
    y = 2.0 * t + rng.normal(scale=3.0, size=t.size)
    res = mann_kendall(y)
    assert res["sen_slope"] > 0
    assert res["z"] > 0
    assert res["p_value"] < 0.05


def test_mann_kendall_detects_decreasing_trend():
    rng = np.random.default_rng(2)
    t = np.arange(40)
    y = -1.5 * t + rng.normal(scale=3.0, size=t.size)
    res = mann_kendall(y)
    assert res["sen_slope"] < 0
    assert res["z"] < 0
    assert res["p_value"] < 0.05


def test_mann_kendall_flat_series_not_significant():
    rng = np.random.default_rng(3)
    y = rng.normal(scale=1.0, size=50)  # no trend
    res = mann_kendall(y)
    assert res["p_value"] > 0.05


# ----------------------------------- GEV ------------------------------------ #
def _gev_sample(n=40, seed=0):
    """A reproducible GEV-like block-maxima sample to fit."""
    from scipy import stats
    return stats.genextreme.rvs(-0.1, loc=30.0, scale=8.0, size=n,
                                random_state=np.random.default_rng(seed))


def test_gev_fit_finite_params():
    bm = _gev_sample(n=40, seed=10)
    fit = fit_gev(bm)
    assert np.isfinite(fit.c)
    assert np.isfinite(fit.loc)
    assert np.isfinite(fit.scale) and fit.scale > 0
    assert np.isfinite(fit.nll)
    assert fit.n == 40


def test_gev_return_levels_monotonic():
    bm = _gev_sample(n=50, seed=11)
    fit = fit_gev(bm)
    z20 = gev_return_level(fit, 20)
    z50 = gev_return_level(fit, 50)
    z100 = gev_return_level(fit, 100)
    assert np.isfinite(z20) and np.isfinite(z50) and np.isfinite(z100)
    assert z20 < z50 < z100  # longer return period -> higher level


# ------------------------------- Bootstrap CI ------------------------------- #
def test_bootstrap_ci_brackets_point():
    bm = _gev_sample(n=45, seed=12)
    out = bootstrap_return_levels(bm, return_periods=(20, 50, 100),
                                  n_boot=60, seed=3)
    for T in (20, 50, 100):
        d = out[T]
        assert np.isfinite(d["point"])
        assert np.isfinite(d["lo"]) and np.isfinite(d["hi"])
        assert d["lo"] <= d["point"] <= d["hi"]
        assert d["n_ok"] >= 2


def test_bootstrap_ci_widens_with_return_period():
    bm = _gev_sample(n=45, seed=13)
    out = bootstrap_return_levels(bm, return_periods=(20, 100),
                                  n_boot=80, seed=4)
    w20 = out[20]["hi"] - out[20]["lo"]
    w100 = out[100]["hi"] - out[100]["lo"]
    # Tail return levels are less certain -> the 100-yr CI is at least as wide.
    assert w100 >= w20


# ------------------------------- Helpers/gate ------------------------------- #
def test_empirical_return_level_monotonic():
    bm = _gev_sample(n=60, seed=14)
    r20 = empirical_return_level(bm, 20)
    r50 = empirical_return_level(bm, 50)
    assert np.isfinite(r20) and np.isfinite(r50)
    assert r50 >= r20


def test_lag1_autocorr_range():
    rng = np.random.default_rng(5)
    y = rng.normal(size=100)
    r = lag1_autocorr(y)
    assert -1.0 <= r <= 1.0


def test_validate_flags_significant_trend():
    # Block maxima with a strong trend -> stationarity warning fires.
    rng = np.random.default_rng(6)
    t = np.arange(30)
    bm = 30.0 + 1.0 * t + rng.normal(scale=2.0, size=t.size)
    fit = fit_gev(bm)
    mk = mann_kendall(bm)
    val = validate(bm, fit, mk, n_tests=3)
    assert isinstance(val.passed, bool)
    # A clear trend should produce at least one warning (stationarity violated).
    assert len(val.warnings) >= 1


def test_pipeline_quick_run_smoke():
    """End-to-end quick run returns a well-formed dict fast."""
    from after.agentic_pipeline import run
    res = run(quick=True, seed=1)
    assert "gev" in res and "return_levels" in res
    for T in (20, 50, 100):
        assert T in res["return_levels"]
    assert res["mk_annual_maxima"]["n"] >= 20
