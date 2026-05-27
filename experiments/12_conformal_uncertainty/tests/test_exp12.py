"""FAST unit tests for Experiment 12 (conformal uncertainty).

CPU-only, no torch, no network. The whole suite is a handful of ridge fits on a
short synthetic series and finishes in well under 30 s (typically a couple of
seconds of test logic plus one-time sklearn/scipy import).

What is asserted (the honest calibration claims):
- Conformal empirical test coverage is within a tolerance of nominal on synthetic
  data (the finite-sample marginal-coverage guarantee, checked empirically).
- In the heteroscedastic regime, conformal's calibration gap is no worse than
  normal-theory's, and strictly better when averaged over levels (the headline win).
- All interval widths are positive and finite; bands are ordered (upper >= lower).
- The conformal quantile applies the finite-sample (n+1) correction correctly.

Run from the REPO ROOT:
    python -m pytest experiments/12_conformal_uncertainty/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root (for `import common`) and the experiment dir (for before/after pkgs).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import daily_temperature  # noqa: E402
from forecaster import build_lag_features, fit_point_forecaster  # noqa: E402
from before.normal_theory import (  # noqa: E402
    normal_z, normal_theory_intervals, run_before,
)
from after.conformal import (  # noqa: E402
    conformal_quantile, split_conformal, normalized_conformal, run_after,
)

LEVELS = (0.80, 0.90, 0.95)
# Tolerance for the empirical-vs-nominal marginal coverage check. Finite test sets
# (~hundreds of points) and weak serial dependence make exact coverage impossible;
# this band is generous enough to be non-flaky yet tight enough to be meaningful.
COVERAGE_TOL = 0.06
# Heteroscedastic regime (real generator knobs) used for the calibration-gap test.
HETERO = dict(heat_extreme_rate=0.06, warming_c_per_decade=0.6)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hetero_split():
    """Short heteroscedastic series, chronological 60/20/20 train/cal/test."""
    df = daily_temperature(n_years=8, seed=0, **HETERO)
    n = len(df)
    tr = df.iloc[: int(0.6 * n)]["t2m"]
    cal = df.iloc[int(0.6 * n): int(0.8 * n)]["t2m"]
    te = df.iloc[int(0.8 * n):]["t2m"]
    return tr, cal, te


def _coverage(y, lo, hi):
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


# ----------------------------- forecaster --------------------------------- #
def test_build_lag_features_shapes_and_no_leakage():
    s = daily_temperature(n_years=2, seed=1)["t2m"]
    X, y, idx = build_lag_features(s, lookback=7)
    assert X.shape[0] == y.shape[0] == len(idx)
    assert X.shape[1] == 7 + 2  # 7 lags + sin/cos
    assert X.shape[0] == len(s) - 7
    assert np.isfinite(X).all() and np.isfinite(y).all()
    # First target uses lags from the first 7 observations, predicts the 8th value.
    assert y[0] == pytest.approx(s.to_numpy()[7])


def test_point_forecaster_predicts_finite():
    s = daily_temperature(n_years=2, seed=2)["t2m"]
    X, y, _ = build_lag_features(s, lookback=7)
    model = fit_point_forecaster(X, y, alpha=1.0)
    pred = model.predict(X)
    assert pred.shape == y.shape and np.isfinite(pred).all()


# ----------------------------- normal-theory (BEFORE) --------------------- #
def test_normal_z_known_values():
    assert normal_z(0.95) == pytest.approx(1.959964, abs=1e-3)
    assert normal_z(0.90) == pytest.approx(1.644854, abs=1e-3)
    assert normal_z(0.80) == pytest.approx(1.281552, abs=1e-3)


def test_normal_theory_intervals_positive_width():
    pred = np.array([10.0, 11.0, 12.0])
    lo, hi, half = normal_theory_intervals(pred, sigma_hat=2.0, level=0.90)
    assert np.all(hi > lo)
    assert np.all(half > 0) and np.all(np.isfinite(half))
    # Symmetric band centred on the point prediction.
    assert np.allclose((lo + hi) / 2.0, pred)


def test_run_before_bands_finite_and_ordered(hetero_split):
    tr, cal, te = hetero_split
    out = run_before(tr, te, levels=LEVELS)
    assert out["sigma_hat"] > 0
    for lvl in LEVELS:
        b = out["bands"][f"{lvl:.2f}"]
        assert np.all(b["upper"] > b["lower"])
        assert np.all(np.isfinite(b["lower"])) and np.all(np.isfinite(b["upper"]))


# ----------------------------- conformal (AFTER) -------------------------- #
def test_conformal_quantile_finite_sample_correction():
    # With n=9 scores 1..9 at level 0.90: rank = ceil(10*0.90)=9 -> 9th smallest = 9.
    scores = np.arange(1, 10, dtype=float)
    assert conformal_quantile(scores, 0.90) == pytest.approx(9.0)
    # When (n+1)*level exceeds n, the honest answer is +inf (need more cal points).
    assert conformal_quantile(np.array([1.0, 2.0]), 0.95) == float("inf")


def test_split_conformal_band_positive_width():
    point_cal = np.zeros(200)
    y_cal = np.random.default_rng(0).normal(0, 1, 200)
    point_test = np.zeros(50)
    lo, hi, half, Q = split_conformal(point_cal, y_cal, point_test, 0.90)
    assert Q > 0 and np.all(half == Q)
    assert np.all(hi > lo)


def test_normalized_conformal_variable_width():
    rng = np.random.default_rng(0)
    point_cal = np.zeros(300)
    sigma_cal = np.full(300, 1.0)
    y_cal = rng.normal(0, 1, 300)
    point_test = np.zeros(4)
    sigma_test = np.array([0.5, 1.0, 2.0, 4.0])  # increasing local spread
    lo, hi, half, Q = normalized_conformal(
        point_cal, y_cal, sigma_cal, point_test, sigma_test, 0.90)
    assert Q > 0 and np.all(half > 0) and np.all(np.isfinite(half))
    # Width must increase monotonically with the conditional-spread estimate.
    assert np.all(np.diff(half) > 0)


def test_run_after_bands_finite_positive(hetero_split):
    tr, cal, te = hetero_split
    out = run_after(tr, cal, te, levels=LEVELS)
    assert out["n_cal"] > 0
    for lvl in LEVELS:
        for key in ("bands", "normalized_bands"):
            b = out[key][f"{lvl:.2f}"]
            assert np.all(b["upper"] > b["lower"])
            assert np.all(np.isfinite(b["half_width"]))
            assert np.all(np.asarray(b["half_width"]) > 0)
            assert np.isfinite(b["Q"]) and b["Q"] > 0


# ----------------------------- the headline claims ------------------------ #
def test_conformal_marginal_coverage_near_nominal(hetero_split):
    """Split-conformal empirical test coverage is within tolerance of nominal."""
    tr, cal, te = hetero_split
    out = run_after(tr, cal, te, levels=LEVELS)
    y = np.asarray(out["y_true"], dtype=float)
    for lvl in LEVELS:
        b = out["bands"][f"{lvl:.2f}"]
        cov = _coverage(y, b["lower"], b["upper"])
        assert abs(cov - lvl) <= COVERAGE_TOL, (
            f"split-conformal coverage {cov:.3f} too far from nominal {lvl}")


def test_normalized_conformal_marginal_coverage_near_nominal(hetero_split):
    tr, cal, te = hetero_split
    out = run_after(tr, cal, te, levels=LEVELS)
    y = np.asarray(out["y_true"], dtype=float)
    for lvl in LEVELS:
        b = out["normalized_bands"][f"{lvl:.2f}"]
        cov = _coverage(y, b["lower"], b["upper"])
        assert abs(cov - lvl) <= COVERAGE_TOL, (
            f"normalized-conformal coverage {cov:.3f} too far from nominal {lvl}")


def test_conformal_closer_to_nominal_than_normal_theory(hetero_split):
    """In the heteroscedastic case, conformal closes the calibration gap.

    Checked two ways: (1) averaged over levels, split-conformal's mean absolute gap
    is strictly smaller than normal-theory's; (2) per level, conformal is never much
    worse than normal-theory (within a small slack).
    """
    tr, cal, te = hetero_split
    before = run_before(tr, te, levels=LEVELS)
    after = run_after(tr, cal, te, levels=LEVELS)
    y = np.asarray(after["y_true"], dtype=float)

    gaps_normal, gaps_split = [], []
    for lvl in LEVELS:
        nb = before["bands"][f"{lvl:.2f}"]
        sb = after["bands"][f"{lvl:.2f}"]
        g_n = abs(_coverage(y, nb["lower"], nb["upper"]) - lvl)
        g_s = abs(_coverage(y, sb["lower"], sb["upper"]) - lvl)
        gaps_normal.append(g_n)
        gaps_split.append(g_s)
        # Conformal is never dramatically worse than normal-theory at any level.
        assert g_s <= g_n + 0.03, f"conformal much worse at level {lvl}: {g_s} vs {g_n}"

    # Headline: averaged over levels, conformal is strictly better calibrated.
    assert np.mean(gaps_split) < np.mean(gaps_normal), (
        f"mean gap split {np.mean(gaps_split):.4f} not < normal "
        f"{np.mean(gaps_normal):.4f}")


def test_normal_theory_is_miscalibrated_in_hetero_regime(hetero_split):
    """Sanity: the BEFORE method really is off-nominal here (otherwise no story)."""
    tr, cal, te = hetero_split
    before = run_before(tr, te, levels=LEVELS)
    y = np.asarray(before["y_true"], dtype=float)
    mean_gap = np.mean([
        abs(_coverage(y, before["bands"][f"{lvl:.2f}"]["lower"],
                      before["bands"][f"{lvl:.2f}"]["upper"]) - lvl)
        for lvl in LEVELS
    ])
    assert mean_gap > 0.01, (
        "normal-theory unexpectedly well-calibrated; the heteroscedastic regime "
        "is the premise of the experiment")
