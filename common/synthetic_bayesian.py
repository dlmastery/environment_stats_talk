"""Synthetic hierarchical Normal-Normal data generator for Bayesian experiments.

Use case
--------
A network of ``S`` environmental monitoring stations measures a noisy quantity
(e.g., a daily mean offset of a station's reading from a global field). Each
station ``s`` has an unknown true mean offset ``theta_s`` drawn from a global
distribution ``Normal(mu, tau^2)``. Each observation at that station is then
``y_{s,i} = theta_s + eps_{s,i}`` with ``eps_{s,i} ~ Normal(0, sigma^2)``. This
is the textbook **partial-pooling / hierarchical Normal-Normal** model used to
shrink noisy station estimates toward the network mean — the same workhorse that
underlies multi-station bias correction, small-area estimation, and meta-analysis.

This generator deliberately ships the **ground-truth latent parameters** alongside
the simulated DataFrame so the experiment can compute *honest* coverage and
RMSE-to-truth metrics (which is impossible with real data because the truth is
unknown).

The generator is fully deterministic (a numpy ``Generator`` seeded by ``seed``),
returns small datasets by design (``S <= 40``, ``n_obs <= ~100`` per station),
and never reads from disk or the network. The output schema is::

    DataFrame columns: ['station', 'obs_idx', 'value']
    truth dict        : {'theta': (S,), 'mu': float, 'tau': float, 'sigma': float,
                         'n_per_station': (S,)}

A single integer ``seed`` reproduces both the latent parameters and the noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "HierarchyParams",
    "synthetic_station_offsets",
]


@dataclass
class HierarchyParams:
    """Hyperparameters of the hierarchical Normal-Normal generator.

    All values are *known* to the BEFORE method's closed-form conjugate update
    (a fixed-hyperparameter Bayes update), and are *inferred* by the
    Metropolis-Hastings sampler and the amortized network.
    """

    mu: float = 0.0          # global mean offset (degC, ppb, etc.)
    tau: float = 1.0         # across-station SD of true offsets (partial-pooling scale)
    sigma: float = 0.8       # within-station observation noise SD
    n_per_station_min: int = 8
    n_per_station_max: int = 24


def synthetic_station_offsets(
    n_stations: int = 30,
    seed: int = 0,
    params: Optional[HierarchyParams] = None,
    equal_n: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Simulate a hierarchical Normal-Normal dataset of station offsets.

    Parameters
    ----------
    n_stations : int
        Number of stations ``S`` (kept small: <= 40 by experiment convention).
    seed : int
        Deterministic seed for both the latent draw and the noise.
    params : HierarchyParams, optional
        Generative hyperparameters. Defaults to ``HierarchyParams()``.
    equal_n : bool
        If True, every station has the same ``n_per_station_max`` observations
        (used in tests for clean closed-form checks).

    Returns
    -------
    df : DataFrame
        Long-form, columns ``['station', 'obs_idx', 'value']``.
    truth : dict
        Ground-truth latent parameters and per-station counts.
    """
    if params is None:
        params = HierarchyParams()
    if n_stations < 1:
        raise ValueError("n_stations must be >= 1")
    if params.tau <= 0 or params.sigma <= 0:
        raise ValueError("tau and sigma must be positive")

    rng = np.random.default_rng(seed)

    # Latent per-station true offsets theta_s ~ N(mu, tau^2)
    theta = rng.normal(params.mu, params.tau, size=n_stations)

    # Per-station observation count.
    if equal_n:
        n_per = np.full(n_stations, params.n_per_station_max, dtype=int)
    else:
        n_per = rng.integers(
            params.n_per_station_min,
            params.n_per_station_max + 1,
            size=n_stations,
        ).astype(int)

    rows = []
    for s in range(n_stations):
        # Per-station observations y_{s,i} = theta_s + eps_{s,i}
        eps = rng.normal(0.0, params.sigma, size=int(n_per[s]))
        ys = theta[s] + eps
        for i, y in enumerate(ys):
            rows.append((s, i, float(y)))

    df = pd.DataFrame(rows, columns=["station", "obs_idx", "value"])

    truth = {
        "theta": theta.copy(),
        "mu": float(params.mu),
        "tau": float(params.tau),
        "sigma": float(params.sigma),
        "n_per_station": n_per.copy(),
    }
    return df, truth
