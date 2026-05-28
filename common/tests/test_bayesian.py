"""FAST unit tests for the synthetic hierarchical Normal-Normal generator.

Run from the repo root:  python -m pytest common/tests/test_bayesian.py -q

Guards: deterministic, well-shaped, statistically consistent with the requested
hyperparameters (means/SDs in the expected range), and exposes the ground-truth
latent parameters needed for honest coverage evaluation in Experiment 09.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Import directly from the submodule (intentionally NOT in common/__init__.py).
from common.synthetic_bayesian import HierarchyParams, synthetic_station_offsets


def test_schema_and_determinism():
    a, ta = synthetic_station_offsets(n_stations=10, seed=3)
    b, tb = synthetic_station_offsets(n_stations=10, seed=3)
    assert list(a.columns) == ["station", "obs_idx", "value"]
    assert np.allclose(a["value"].to_numpy(), b["value"].to_numpy())
    assert np.allclose(ta["theta"], tb["theta"])
    assert ta["mu"] == tb["mu"] and ta["tau"] == tb["tau"] and ta["sigma"] == tb["sigma"]


def test_truth_contains_required_fields():
    df, truth = synthetic_station_offsets(n_stations=8, seed=1)
    for k in ("theta", "mu", "tau", "sigma", "n_per_station"):
        assert k in truth
    assert truth["theta"].shape == (8,)
    assert truth["n_per_station"].shape == (8,)
    assert df.shape[0] == int(truth["n_per_station"].sum())


def test_per_station_means_track_truth():
    """Per-station sample means should be close to the true theta (within the
    Normal SE) — sanity check that the generator implements y = theta + eps."""
    params = HierarchyParams(mu=0.0, tau=1.5, sigma=0.5,
                             n_per_station_min=50, n_per_station_max=50)
    df, truth = synthetic_station_offsets(n_stations=20, seed=42,
                                          params=params, equal_n=True)
    ybar = df.groupby("station")["value"].mean().to_numpy()
    # SE = sigma / sqrt(n) = 0.5 / sqrt(50) ~ 0.07; allow 6 SE slack
    assert np.allclose(ybar, truth["theta"], atol=6 * 0.5 / np.sqrt(50))


def test_between_station_spread_matches_tau():
    """SD of true thetas across many stations should be close to tau."""
    params = HierarchyParams(mu=0.0, tau=1.0, sigma=0.5)
    _, truth = synthetic_station_offsets(n_stations=40, seed=7, params=params)
    sd_theta = float(np.std(truth["theta"], ddof=1))
    # tau = 1, sampling SD of SD with S=40 is roughly tau/sqrt(2(S-1)) ~ 0.11
    assert abs(sd_theta - 1.0) < 0.35


def test_within_station_noise_matches_sigma():
    params = HierarchyParams(mu=0.0, tau=2.0, sigma=0.8,
                             n_per_station_min=40, n_per_station_max=40)
    df, truth = synthetic_station_offsets(n_stations=15, seed=11,
                                          params=params, equal_n=True)
    # Center each station's obs at its true theta and pool the residuals
    merged = df.merge(
        pd.DataFrame({"station": np.arange(15), "theta": truth["theta"]}),
        on="station",
    )
    resid = merged["value"].to_numpy() - merged["theta"].to_numpy()
    assert abs(resid.std(ddof=1) - 0.8) < 0.1
