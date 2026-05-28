"""FAST unit tests for Experiment 09 (Bayesian hierarchical vs amortized inference).

Run from REPO ROOT:
    python -m pytest experiments/09_bayesian_vs_amortized/tests -q

These are CPU-only, deterministic, and tight (well under 30s total). They check:
    * the closed-form conjugate posterior is correct on a hand-made tiny case;
    * the MH sampler produces finite draws and ~nominal coverage of true theta;
    * the amortized network trains on a few hundred simulated datasets and gives
      finite, ~nominal-coverage intervals;
    * the per-dataset SCORING of the amortized model is strictly faster than
      a tiny MCMC fit (the whole "amortized inference" headline).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path setup so imports mirror the run script.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from common.synthetic_bayesian import HierarchyParams, synthetic_station_offsets  # noqa: E402
from before.classical_bayes import (  # noqa: E402
    station_sufficient_stats,
    conjugate_posterior_known_hyper,
    mh_sampler,
    summarize_mh,
)
from after.amortized import (  # noqa: E402
    AmortizedConfig, train_amortized, score_amortized, build_features_from_suff,
    N_FEATURES,
)

_CPU = torch.device("cpu")


# ----------------------------- helpers ------------------------------------ #
def _coverage(post: pd.DataFrame, theta_true: np.ndarray) -> float:
    lo = post["lo95"].to_numpy(); hi = post["hi95"].to_numpy()
    return float(np.mean((theta_true >= lo) & (theta_true <= hi)))


# ----------------------------- conjugate ---------------------------------- #
def test_sufficient_stats_shapes():
    df, _ = synthetic_station_offsets(n_stations=6, seed=0)
    s = station_sufficient_stats(df)
    assert list(s.columns) == ["n", "ybar", "svar"]
    assert len(s) == 6
    assert (s["n"] > 0).all()
    assert np.isfinite(s.to_numpy()).all()


def test_conjugate_matches_handcomputed_value():
    """One station, one obs: posterior is the textbook precision-weighted mix."""
    df = pd.DataFrame({"station": [0, 0, 0], "obs_idx": [0, 1, 2],
                       "value": [1.0, 1.2, 0.8]})
    s = station_sufficient_stats(df)
    # mu=0, tau=2, sigma=1: prec = 1/4 + 3/1 = 3.25; var = 1/3.25
    # mean = (0/4 + 3*1.0/1)/3.25 = 3.0/3.25
    post = conjugate_posterior_known_hyper(s, mu=0.0, tau=2.0, sigma=1.0)
    assert post.loc[0, "post_mean"] == pytest.approx(3.0 / 3.25, abs=1e-10)
    assert post.loc[0, "post_sd"] == pytest.approx(np.sqrt(1.0 / 3.25), abs=1e-10)


def test_conjugate_coverage_at_known_hyper():
    """At KNOWN hyperparameters the closed-form CIs should hit ~95% coverage."""
    params = HierarchyParams(mu=0.0, tau=1.0, sigma=0.6,
                             n_per_station_min=10, n_per_station_max=20)
    df, truth = synthetic_station_offsets(n_stations=40, seed=0, params=params)
    s = station_sufficient_stats(df)
    post = conjugate_posterior_known_hyper(s, mu=params.mu, tau=params.tau,
                                           sigma=params.sigma)
    cov = _coverage(post, truth["theta"])
    # Generous bound (single dataset, S=40): nominal 0.95 with binomial noise
    assert 0.80 <= cov <= 1.0
    assert np.isfinite(post.to_numpy()).all()


# ----------------------------- MH sampler --------------------------------- #
def test_mh_runs_and_returns_finite_draws():
    params = HierarchyParams(mu=0.2, tau=1.0, sigma=0.7)
    df, truth = synthetic_station_offsets(n_stations=10, seed=1, params=params)
    s = station_sufficient_stats(df)
    mh = mh_sampler(s, n_iter=500, n_burn=150, seed=0)
    assert mh.theta.shape == (350, 10)
    assert np.isfinite(mh.theta).all()
    assert np.isfinite(mh.mu).all() and (mh.tau > 0).all() and (mh.sigma > 0).all()
    # Reasonable acceptance per block (rough RW guidance)
    for k, v in mh.accept_rate.items():
        assert 0.05 <= v <= 0.95, f"degenerate acceptance for {k}: {v}"


def test_mh_coverage_close_to_nominal():
    """MH on the full hierarchical posterior should give ~95% coverage of
    the *true* station effects on a single moderately-sized dataset."""
    params = HierarchyParams(mu=0.0, tau=1.0, sigma=0.6,
                             n_per_station_min=10, n_per_station_max=20)
    df, truth = synthetic_station_offsets(n_stations=25, seed=2, params=params)
    s = station_sufficient_stats(df)
    mh = mh_sampler(s, n_iter=1500, n_burn=500, seed=0)
    post = summarize_mh(mh, station_index=s.index)
    cov = _coverage(post, truth["theta"])
    # Loose nominal bound (single dataset of 25 stations; binomial noise on 0.95)
    assert 0.75 <= cov <= 1.0


# ----------------------------- amortized ---------------------------------- #
def test_features_shape():
    df, _ = synthetic_station_offsets(n_stations=12, seed=0)
    s = station_sufficient_stats(df)
    feats = build_features_from_suff(s)
    assert feats.shape == (12, N_FEATURES)
    assert np.isfinite(feats).all()


@pytest.fixture(scope="module")
def trained_amortized():
    """Tiny amortized network trained on a few hundred simulated datasets."""
    cfg = AmortizedConfig(
        hidden=32, n_layers=2,
        n_train_datasets=300, batch_datasets=32, n_epochs=1,
        s_min=8, s_max=20,
        n_per_station_min=6, n_per_station_max=18,
        seed=0,
    )
    model, stats = train_amortized(cfg, device=_CPU)
    return model, stats


def test_amortized_scoring_finite(trained_amortized):
    model, stats = trained_amortized
    params = HierarchyParams(mu=0.3, tau=1.1, sigma=0.7,
                             n_per_station_min=8, n_per_station_max=18)
    df, _ = synthetic_station_offsets(n_stations=15, seed=11, params=params)
    s = station_sufficient_stats(df)
    post = score_amortized(model, s, stats, device=_CPU)
    assert np.isfinite(post.to_numpy()).all()
    assert (post["post_sd"] > 0).all()
    assert (post["hi95"] > post["lo95"]).all()


def test_amortized_coverage_within_tolerance(trained_amortized):
    """Amortized intervals should hit ~95% coverage within a tolerance (a tiny
    network trained on 300 sims won't be perfectly calibrated)."""
    model, stats = trained_amortized
    params = HierarchyParams(mu=0.0, tau=1.0, sigma=0.6,
                             n_per_station_min=10, n_per_station_max=18)
    # Average coverage across a few fresh datasets to reduce binomial noise.
    covs = []
    for seed in (101, 202, 303, 404, 505):
        df, truth = synthetic_station_offsets(n_stations=20, seed=seed, params=params)
        s = station_sufficient_stats(df)
        post = score_amortized(model, s, stats, device=_CPU)
        covs.append(_coverage(post, truth["theta"]))
    mean_cov = float(np.mean(covs))
    # Generous: amortized at this tiny training budget targets ~0.95 but may
    # be a touch under/over. Require coverage in [0.70, 1.0].
    assert 0.70 <= mean_cov <= 1.0, f"mean coverage out of band: {mean_cov:.3f}"


def test_amortized_scoring_faster_than_mh_fit(trained_amortized):
    """Headline: per-new-dataset SCORING of the amortized model is *much* faster
    than even a tiny MH fit (the whole point of amortized inference)."""
    model, stats = trained_amortized
    params = HierarchyParams(mu=0.0, tau=1.0, sigma=0.6)
    df, _ = synthetic_station_offsets(n_stations=15, seed=77, params=params)
    s = station_sufficient_stats(df)

    # warm up amortized forward pass
    _ = score_amortized(model, s, stats, device=_CPU)

    t0 = time.time()
    _ = score_amortized(model, s, stats, device=_CPU)
    t_score = time.time() - t0

    t0 = time.time()
    _ = mh_sampler(s, n_iter=400, n_burn=100, seed=0)
    t_mh = time.time() - t0

    # Amortized scoring should be at LEAST 10x faster than even a tiny MH fit.
    assert t_score < t_mh, f"score {t_score:.4f}s >= mh {t_mh:.4f}s"
    assert t_score * 10 < t_mh, (
        f"amortized score not >=10x faster than MH: score={t_score:.4f}s "
        f"vs mh={t_mh:.4f}s"
    )


# ----------------------------- run-harness smoke -------------------------- #
def test_run_quick_writes_artifacts(tmp_path, monkeypatch):
    """End-to-end smoke: run the orchestrator in --quick mode and verify the
    advertised artifacts land in results/ with finite numbers."""
    import importlib
    import run_before_after as rba
    importlib.reload(rba)

    # Redirect RESULTS_DIR into tmp_path to avoid clobbering committed results.
    monkeypatch.setattr(rba, "RESULTS_DIR", tmp_path)

    args = rba.parse_args(["--quick"])
    res = rba.run(args)
    for k in ("before_closed_form_known_hyper", "before_mh", "after_amortized",
              "wall_time_sec"):
        assert k in res
    # Numeric sanity
    assert np.isfinite(res["before_mh"]["coverage_95"])
    assert np.isfinite(res["after_amortized"]["coverage_95"])
    for f in ("metrics.json", "summary.md",
              "posterior_intervals.png", "before_after_bars.png"):
        assert (tmp_path / f).exists(), f"missing artifact: {f}"
