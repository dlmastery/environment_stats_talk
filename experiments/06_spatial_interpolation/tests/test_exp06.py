"""FAST unit tests for Experiment 06 (spatial interpolation).

CPU-only and tiny (a few hundred stations on a small grid, modest tree ensembles),
so the whole suite runs in well under 30 s. They check that the coordinate-only
BEFORE interpolators and the covariate-aware AFTER model both run and return finite
metrics, that the empirical variogram is sane, and — the headline of this
experiment — that on a held-out grid the covariate-aware ML attains RMSE no worse
than coordinate-only kriging (RMSE_ml <= RMSE_kriging).

Run from the REPO ROOT:
    python -m pytest experiments/06_spatial_interpolation/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root (for `import common`) and the experiment dir (for before/after).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.synthetic_spatial import synthetic_spatial_field  # noqa: E402
from common.metrics import rmse, mae  # noqa: E402
from before.kriging import (  # noqa: E402
    idw_predict, fit_kriging, kriging_predict, empirical_variogram, run_before,
)
from after.ml_interp import build_features, fit_ml, run_after, FEATURE_NAMES  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    # Small but enough to demonstrate the covariate win; fast on CPU.
    return synthetic_spatial_field(n_points=140, seed=0, grid_res=24)


# ----------------------------- BEFORE: IDW -------------------------------- #
def test_idw_shapes_and_exactness(dataset):
    ds = dataset
    yp = idw_predict(ds.train_coords, ds.train_value, ds.grid_coords)
    assert yp.shape == (ds.grid_coords.shape[0],)
    assert np.isfinite(yp).all()
    # IDW at a training location returns (essentially) that station's value.
    yp_at_stations = idw_predict(ds.train_coords, ds.train_value, ds.train_coords)
    assert np.allclose(yp_at_stations, ds.train_value, atol=1e-6)


# ----------------------------- BEFORE: kriging ---------------------------- #
def test_kriging_fits_and_predicts_with_uncertainty(dataset):
    ds = dataset
    gp = fit_kriging(ds.train_coords, ds.train_value, n_restarts=1, seed=0)
    mean, std = kriging_predict(gp, ds.grid_coords)
    assert mean.shape == std.shape == (ds.grid_coords.shape[0],)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    # kriging variance is non-negative; it is the principled uncertainty surface
    assert (std >= 0).all()
    assert std.max() > 0


def test_empirical_variogram_increases(dataset):
    ds = dataset
    centres, gamma, counts = empirical_variogram(
        ds.train_coords, ds.train_value, n_bins=12)
    good = np.isfinite(gamma)
    assert good.sum() >= 4
    assert (counts[good] > 0).all()
    # there IS spatial structure: short-lag semivariance < long-lag semivariance
    g = gamma[good]
    assert g[0] < g[-1]


def test_run_before_metrics_finite(dataset):
    ds = dataset
    res = run_before(ds, n_restarts=1, seed=0)
    assert "idw" in res and "kriging" in res
    for name in ("idw", "kriging"):
        assert np.isfinite(res[name]["rmse"])
        assert np.isfinite(res[name]["mae"])
    assert "y_std" in res["kriging"]  # uncertainty surface present


# ----------------------------- AFTER: ML ---------------------------------- #
def test_build_features_shape(dataset):
    ds = dataset
    X = build_features(ds.train_coords, ds.train_covariate, ds.params.domain)
    assert X.shape == (ds.train_coords.shape[0], len(FEATURE_NAMES))
    assert np.isfinite(X).all()


def test_fit_ml_runs_rf_and_gbm(dataset):
    ds = dataset
    X = build_features(ds.train_coords, ds.train_covariate, ds.params.domain)
    for model in ("rf", "gbm"):
        reg = fit_ml(X, ds.train_value, model=model, n_estimators=60, seed=0)
        pred = reg.predict(X)
        assert pred.shape == (X.shape[0],)
        assert np.isfinite(pred).all()


def test_run_after_metrics_finite_and_uses_covariate(dataset):
    ds = dataset
    res = run_after(ds, model="rf", n_estimators=120, seed=0)
    assert np.isfinite(res["rmse"]) and np.isfinite(res["mae"])
    assert len(res["y_pred"]) == ds.grid_coords.shape[0]
    imp = res["feature_importance"]
    assert imp  # tree model exposes importances
    # the covariate (or a covariate interaction) is among the most important features
    cov_share = imp["covariate"] + imp["x*cov"] + imp["y*cov"] + imp["covariate^2"]
    coord_share = imp["x"] + imp["y"] + imp["x*y"]
    assert cov_share > coord_share


# ----------------------------- headline: AFTER <= BEFORE RMSE -------------- #
def test_ml_beats_kriging_on_holdout(dataset):
    """The whole point of this experiment: a covariate-aware ML model attains RMSE no
    worse than coordinate-only kriging on the held-out grid (in practice clearly
    lower). Determinism makes this a stable assertion."""
    ds = dataset
    before = run_before(ds, n_restarts=1, seed=0)
    after = run_after(ds, model="rf", n_estimators=200, seed=0)

    rmse_krig = before["kriging"]["rmse"]
    rmse_ml = after["rmse"]
    assert np.isfinite(rmse_krig) and np.isfinite(rmse_ml)
    assert rmse_ml <= rmse_krig          # AFTER wins (or at worst ties)
    # and kriging still beats naive IDW (coords-only methods are not strawmen)
    assert before["kriging"]["rmse"] < before["idw"]["rmse"]
