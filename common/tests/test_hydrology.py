"""FAST unit tests for the synthetic rainfall-runoff generator.

Run from the repo root:  python -m pytest common/tests/test_hydrology.py -q

These guarantee the synthetic catchment is deterministic, well-shaped, physically
sane (non-negative streamflow, many dry precip days), and that streamflow actually
RESPONDS to precipitation with the nonlinear, antecedent-moisture memory the
experiment depends on (so a linear model underfits and an LSTM can win).
"""
from __future__ import annotations

import numpy as np

# Import the generator from the submodule directly (it is intentionally NOT in
# common/__init__.py).
from common.synthetic_hydrology import synthetic_catchment, CatchmentParams


def test_schema_and_determinism():
    a = synthetic_catchment(n_years=3, seed=7)
    b = synthetic_catchment(n_years=3, seed=7)
    assert a.shape == (3 * 365, 4)
    assert list(a.columns) == ["precip", "temperature", "pet", "streamflow"]
    assert a.index.name == "date"
    assert np.allclose(a["streamflow"].to_numpy(), b["streamflow"].to_numpy())


def test_streamflow_nonnegative_and_finite():
    df = synthetic_catchment(n_years=4, seed=1)
    q = df["streamflow"].to_numpy()
    assert (q >= 0).all()
    assert np.isfinite(df.to_numpy()).all()
    assert q.max() > 0  # there is flow


def test_precip_intermittent_and_pet_nonneg():
    df = synthetic_catchment(n_years=4, seed=2)
    p = df["precip"].to_numpy()
    assert (p >= 0).all()
    dry_fraction = float(np.mean(p == 0))
    assert 0.3 < dry_fraction < 0.95   # many dry days, but not all
    assert (df["pet"].to_numpy() >= 0).all()


def test_streamflow_responds_to_precipitation():
    """Antecedent (multi-day) precipitation should drive streamflow: the 7-day
    precip sum correlates more strongly with flow than same-day rain — the
    nonlinear-memory signature the LSTM exploits and the linear model misses."""
    df = synthetic_catchment(n_years=8, seed=0)
    q = df["streamflow"].to_numpy()
    p_same = df["precip"].to_numpy()
    p_7d = df["precip"].rolling(7).sum().fillna(0.0).to_numpy()
    r_same = float(np.corrcoef(q, p_same)[0, 1])
    r_7d = float(np.corrcoef(q, p_7d)[0, 1])
    assert r_7d > 0.2                  # streamflow clearly responds to recent rain
    assert r_7d > r_same               # memory: antecedent rain matters more


def test_more_nonlinearity_lowers_linear_correlation():
    """A higher saturation exponent (beta) makes the response more nonlinear, which
    should not increase the same-day linear correlation — sanity check on the knob."""
    df_lin = synthetic_catchment(n_years=6, seed=3,
                                 params=CatchmentParams(beta=1.0))
    df_nl = synthetic_catchment(n_years=6, seed=3,
                                params=CatchmentParams(beta=4.0))
    r_lin = abs(np.corrcoef(df_lin["streamflow"], df_lin["precip"])[0, 1])
    r_nl = abs(np.corrcoef(df_nl["streamflow"], df_nl["precip"])[0, 1])
    assert r_nl <= r_lin + 0.05        # stronger nonlinearity, not more linear signal
