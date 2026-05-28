"""FAST unit tests for the synthetic SDM generator.

Run from the repo root:  python -m pytest common/tests/test_sdm.py -q

These guarantee the synthetic species-distribution dataset is deterministic,
well-shaped, has both presence and background classes, and exposes a TRUE
suitability surface that the experiments score against. The generator is
imported from the submodule directly (it is intentionally NOT in
common/__init__.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.synthetic_sdm import (
    SDMDataset, SDMParams, synthetic_sdm_dataset,
)


def test_shapes_and_types():
    ds = synthetic_sdm_dataset(n_pres=120, n_bg=600, seed=0)
    assert isinstance(ds, SDMDataset)
    # grid
    ny, nx = ds.grid_shape
    assert ds.temperature_field.shape == (ny, nx)
    assert ds.precipitation_field.shape == (ny, nx)
    assert ds.elevation_field.shape == (ny, nx)
    assert ds.suitability_field.shape == (ny, nx)
    assert ds.grid_x.shape == (nx,)
    assert ds.grid_y.shape == (ny,)
    # flat tables
    assert isinstance(ds.grid_df, pd.DataFrame)
    assert isinstance(ds.presence_df, pd.DataFrame)
    assert isinstance(ds.background_df, pd.DataFrame)
    assert isinstance(ds.records_df, pd.DataFrame)
    for col in ("x", "y", "temperature", "precipitation", "elevation"):
        assert col in ds.records_df.columns
    assert "y_label" in ds.records_df.columns
    assert "suitability" in ds.grid_df.columns


def test_determinism():
    a = synthetic_sdm_dataset(n_pres=80, n_bg=400, seed=7)
    b = synthetic_sdm_dataset(n_pres=80, n_bg=400, seed=7)
    assert np.allclose(a.suitability_field, b.suitability_field)
    assert np.allclose(a.temperature_field, b.temperature_field)
    assert np.allclose(a.background_df["x"].to_numpy(),
                       b.background_df["x"].to_numpy())
    assert len(a.presence_df) == len(b.presence_df)
    # different seed -> different field
    c = synthetic_sdm_dataset(n_pres=80, n_bg=400, seed=8)
    assert not np.allclose(a.suitability_field, c.suitability_field)


def test_both_classes_present():
    ds = synthetic_sdm_dataset(n_pres=100, n_bg=500, seed=0)
    classes = set(ds.records_df["y_label"].unique().tolist())
    assert classes == {0, 1}, f"expected both presence and background, got {classes}"
    assert (ds.records_df["y_label"] == 1).sum() > 0
    assert (ds.records_df["y_label"] == 0).sum() > 0
    # background count is exact; presence count is approximately requested
    assert len(ds.background_df) == 500
    assert len(ds.presence_df) > 0 and len(ds.presence_df) <= 100


def test_all_finite_and_suitability_in_unit_range():
    ds = synthetic_sdm_dataset(n_pres=80, n_bg=300, seed=2)
    for arr in (ds.temperature_field, ds.precipitation_field,
                ds.elevation_field, ds.suitability_field):
        assert np.isfinite(arr).all()
    for col in ("temperature", "precipitation", "elevation"):
        assert np.isfinite(ds.records_df[col].to_numpy()).all()
    # suitability is a probability-like surface
    assert ds.suitability_field.min() >= 0.0
    assert ds.suitability_field.max() <= 1.0
    # the niche is genuine: presence rows live where suitability is higher than
    # at uniformly-sampled background rows (sanity for the rejection sampler).
    if len(ds.presence_df) > 10:
        # interpolate suitability at presence and background locations from the grid
        from common.synthetic_sdm import _interp_field
        suit_pres = _interp_field(ds.grid_x, ds.grid_y, ds.suitability_field,
                                  ds.presence_df["x"].to_numpy(),
                                  ds.presence_df["y"].to_numpy())
        suit_bg = _interp_field(ds.grid_x, ds.grid_y, ds.suitability_field,
                                ds.background_df["x"].to_numpy(),
                                ds.background_df["y"].to_numpy())
        assert suit_pres.mean() > suit_bg.mean(), (
            "presence sites should have higher mean true suitability than "
            "uniformly-sampled background")


def test_grid_df_consistent_with_fields():
    ds = synthetic_sdm_dataset(n_pres=60, n_bg=200, seed=3)
    ny, nx = ds.grid_shape
    # grid_df has n_grid rows
    assert len(ds.grid_df) == ny * nx
    # the flattened temperature field matches grid_df['temperature']
    assert np.allclose(ds.grid_df["temperature"].to_numpy(),
                       ds.temperature_field.ravel())
    assert np.allclose(ds.grid_df["suitability"].to_numpy(),
                       ds.suitability_field.ravel())


def test_records_df_is_concat_of_presence_and_background():
    ds = synthetic_sdm_dataset(n_pres=70, n_bg=350, seed=5)
    assert len(ds.records_df) == len(ds.presence_df) + len(ds.background_df)
    # column set is consistent (presence_df + background_df both have y_label)
    expected_cols = {"x", "y", "temperature", "precipitation",
                     "elevation", "y_label"}
    assert expected_cols.issubset(set(ds.records_df.columns))
