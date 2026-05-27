"""FAST unit tests for the synthetic spatial-field generator.

Run from the repo root:  python -m pytest common/tests/test_spatial.py -q

These guarantee the synthetic field is deterministic, well-shaped, and — the crux of
Experiment 06 — that the covariate (elevation) is GENUINELY informative about the
value (so coordinate-only interpolation leaves real skill on the table and a
covariate-aware model can do better). The generator is imported from the submodule
directly (it is intentionally NOT in common/__init__.py).
"""
from __future__ import annotations

import numpy as np

from common.synthetic_spatial import (
    synthetic_spatial_field, SpatialFieldParams, SpatialDataset,
)


def test_shapes_and_types():
    ds = synthetic_spatial_field(n_points=120, seed=0, grid_res=30)
    assert isinstance(ds, SpatialDataset)
    assert ds.train_coords.shape == (120, 2)
    assert ds.train_covariate.shape == (120,)
    assert ds.train_value.shape == (120,)
    assert ds.train_value_clean.shape == (120,)
    assert ds.grid_shape == (30, 30)
    assert ds.grid_coords.shape == (900, 2)
    assert ds.grid_covariate.shape == (900,)
    assert ds.grid_value.shape == (900,)
    assert ds.cov_field.shape == (30, 30)
    assert ds.grid_x.shape == (30,) and ds.grid_y.shape == (30,)


def test_determinism():
    a = synthetic_spatial_field(n_points=100, seed=7, grid_res=25)
    b = synthetic_spatial_field(n_points=100, seed=7, grid_res=25)
    assert np.allclose(a.train_value, b.train_value)
    assert np.allclose(a.grid_value, b.grid_value)
    assert np.allclose(a.train_covariate, b.train_covariate)
    # a different seed gives a different field
    c = synthetic_spatial_field(n_points=100, seed=8, grid_res=25)
    assert not np.allclose(a.grid_value, c.grid_value)


def test_all_finite_and_grid_axes_ordered():
    ds = synthetic_spatial_field(n_points=100, seed=1, grid_res=20)
    for arr in (ds.train_coords, ds.train_value, ds.train_covariate,
                ds.grid_coords, ds.grid_value, ds.grid_covariate, ds.cov_field):
        assert np.isfinite(arr).all()
    # grid axes are monotonically increasing over the domain
    assert np.all(np.diff(ds.grid_x) > 0)
    assert np.all(np.diff(ds.grid_y) > 0)
    assert ds.grid_x[0] >= 0.0 and ds.grid_x[-1] <= ds.params.domain + 1e-9


def test_cov_field_matches_grid_covariate():
    """The 2-D covariate image must be the grid covariate reshaped consistently."""
    ds = synthetic_spatial_field(n_points=80, seed=2, grid_res=18)
    ny, nx = ds.grid_shape
    assert np.allclose(ds.cov_field, ds.grid_covariate.reshape(ny, nx))


def test_observation_noise_is_small_perturbation():
    ds = synthetic_spatial_field(n_points=150, seed=3, grid_res=20)
    resid = ds.train_value - ds.train_value_clean
    # noise std should be close to the configured noise_sd, and much smaller than the
    # field's own spread (so the signal dominates).
    assert np.std(resid) < 0.6 * ds.params.noise_sd + ds.params.noise_sd
    assert np.std(resid) < 0.3 * np.std(ds.train_value)


def test_covariate_is_genuinely_informative():
    """The covariate must explain a large, real fraction of the field's variance.

    This is what makes covariate-aware ML able to beat coordinate-only kriging. We
    check both a strong |correlation| and a high univariate R^2 on the grid truth.
    """
    ds = synthetic_spatial_field(n_points=160, seed=0, grid_res=40)
    cov, val = ds.grid_covariate, ds.grid_value
    r = float(np.corrcoef(cov, val)[0, 1])
    assert abs(r) > 0.5
    coef = np.polyfit(cov, val, 1)
    pred = np.polyval(coef, cov)
    r2 = 1 - np.sum((val - pred) ** 2) / np.sum((val - val.mean()) ** 2)
    assert r2 > 0.4


def test_covariate_has_short_scale_structure():
    """The covariate must vary on SHORTER scales than the smooth GRF, otherwise a
    coordinate-only interpolator could mimic it. Compare the value at nearby station
    pairs: pairs that are spatially close but differ a lot in covariate should also
    differ a lot in value — evidence the covariate carries sub-grid information."""
    ds = synthetic_spatial_field(n_points=200, seed=5, grid_res=20)
    c = ds.train_coords
    v = ds.train_value
    cov = ds.train_covariate
    # all close pairs (within a short radius relative to the GRF length scale)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(len(c), k=1)
    close = d[iu, ju] < 0.4 * ds.params.grf_length
    dv = np.abs(v[iu] - v[ju])[close]
    dcov = np.abs(cov[iu] - cov[ju])[close]
    # among spatially-close pairs, value difference correlates with covariate diff
    if dcov.size > 10 and np.std(dcov) > 0:
        r = float(np.corrcoef(dcov, dv)[0, 1])
        assert r > 0.2


def test_lapse_sign_controls_correlation():
    """Flipping the covariate sensitivity sign flips the value/covariate correlation —
    a sanity check that the covariate truly drives the value."""
    neg = synthetic_spatial_field(n_points=120, seed=4, grid_res=20,
                                  params=SpatialFieldParams(lapse=-7.0))
    pos = synthetic_spatial_field(n_points=120, seed=4, grid_res=20,
                                  params=SpatialFieldParams(lapse=+7.0))
    r_neg = np.corrcoef(neg.grid_covariate, neg.grid_value)[0, 1]
    r_pos = np.corrcoef(pos.grid_covariate, pos.grid_value)[0, 1]
    assert r_neg < 0 < r_pos
