"""Unit tests for the shared common/ utilities.

Run from the repo root:  python -m pytest common/tests -v
These tests guarantee the synthetic generators are deterministic, well-shaped,
and carry the signal each experiment depends on (trend, extremes, separability).
"""
from __future__ import annotations

import numpy as np
import pytest

from common import (
    metrics, daily_temperature, daily_precipitation, gridded_temperature_field,
    time_split, citizen_comments, gold_interactions,
    multispectral_patches, compute_indices, change_pair, CLASSES,
)


# ----------------------------- metrics -----------------------------------
def test_rmse_mae_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, y) == pytest.approx(0.0)
    assert metrics.mae(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    assert metrics.rmse([0, 0], [1, 1]) == pytest.approx(1.0)
    assert metrics.mae([0, 0], [2, 0]) == pytest.approx(1.0)


def test_anomaly_correlation_perfect_and_anti():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.anomaly_correlation(y, y) == pytest.approx(1.0, abs=1e-6)
    assert metrics.anomaly_correlation(y, -y + 2 * y.mean()) == pytest.approx(-1.0, abs=1e-6)


def test_latitude_weighted_rmse_weights_poles_less():
    n_lat, n_lon = 8, 12
    lats = np.linspace(-80, 80, n_lat)
    truth = np.zeros((n_lat, n_lon))
    # Error only at the highest-latitude row -> down-weighted -> small lat-wtd RMSE.
    err_pole = truth.copy(); err_pole[-1, :] = 10.0
    err_eq = truth.copy(); err_eq[n_lat // 2, :] = 10.0
    r_pole = metrics.latitude_weighted_rmse(truth, err_pole, lats)
    r_eq = metrics.latitude_weighted_rmse(truth, err_eq, lats)
    assert r_pole < r_eq  # the same-magnitude error near the pole counts less


def test_skill_score_error_metric():
    # model error half of reference -> skill 0.5
    assert metrics.skill_score(0.5, 1.0, higher_is_better=False) == pytest.approx(0.5)


def test_classification_report_simple():
    rep = metrics.classification_report_simple([0, 1, 1, 0], [0, 1, 0, 0])
    assert rep["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= rep["macro_f1"] <= 1.0


# ----------------------------- climate -----------------------------------
def test_daily_temperature_deterministic_and_shaped():
    a = daily_temperature(n_years=3, seed=7)
    b = daily_temperature(n_years=3, seed=7)
    assert a.shape == (3 * 365, 1)
    assert list(a.columns) == ["t2m"]
    assert np.allclose(a["t2m"].to_numpy(), b["t2m"].to_numpy())  # deterministic


def test_daily_temperature_has_warming_trend():
    df = daily_temperature(n_years=20, seed=0, warming_c_per_decade=0.5)
    first = df["t2m"].iloc[:365 * 3].mean()
    last = df["t2m"].iloc[-365 * 3:].mean()
    assert last > first  # detectable warming over the record


def test_daily_precipitation_intermittent_and_nonneg():
    df = daily_precipitation(n_years=5, seed=1)
    p = df["precip"].to_numpy()
    assert (p >= 0).all()
    dry_fraction = float(np.mean(p == 0))
    assert 0.4 < dry_fraction < 0.95  # many dry days, but not all
    assert p.max() > np.percentile(p[p > 0], 95)  # heavy upper tail exists


def test_gridded_field_shapes_and_latitude_gradient():
    gf = gridded_temperature_field(n_lat=16, n_lon=32, seed=2)
    assert gf.field.shape == (16, 32)
    assert gf.lats.shape == (16,) and gf.lons.shape == (32,)
    # equatorial band warmer than polar band on average
    eq = gf.field[6:10].mean()
    pole = np.concatenate([gf.field[:2], gf.field[-2:]]).mean()
    assert eq > pole


def test_time_split_no_overlap_and_chronological():
    df = daily_temperature(n_years=4, seed=3)
    tr, va, te = time_split(df, 0.7, 0.15)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.index.max() < va.index.min() < te.index.min()


# --------------------------- biodiversity --------------------------------
def test_citizen_comments_schema_and_determinism():
    c1 = citizen_comments(n=50, seed=0)
    c2 = citizen_comments(n=50, seed=0)
    assert len(c1) == 50
    assert [r["text"] for r in c1] == [r["text"] for r in c2]
    for r in c1:
        assert set(r) == {"id", "text", "interactions"}
        for trip in r["interactions"]:
            assert len(trip) == 3


def test_gold_interactions_subset_of_comment_labels():
    gold = gold_interactions()
    assert len(gold) >= 8
    comments = citizen_comments(n=400, seed=1)
    seen = {t for r in comments for t in r["interactions"]}
    assert seen.issubset(gold)  # every labeled interaction is a valid gold triple


# -------------------------- remote sensing -------------------------------
def test_multispectral_patches_shape_range():
    X, y = multispectral_patches(n=40, size=8, seed=0)
    assert X.shape == (40, 5, 8, 8)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)).issubset(set(range(len(CLASSES))))


def test_compute_indices_ndvi_forest_gt_water():
    Xf, _ = multispectral_patches(n=30, size=8, seed=10)
    feats = compute_indices(Xf)
    assert feats.shape == (30, 4)
    # NDVI in [-1,1]
    assert feats[:, 0].min() >= -1.01 and feats[:, 0].max() <= 1.01


def test_change_pair_labels_and_shapes():
    X0, X1, changed = change_pair(n=50, size=8, seed=0, change_frac=0.4)
    assert X0.shape == X1.shape == (50, 5, 8, 8)
    assert set(np.unique(changed)).issubset({0, 1})
    assert 0 < changed.mean() < 1
