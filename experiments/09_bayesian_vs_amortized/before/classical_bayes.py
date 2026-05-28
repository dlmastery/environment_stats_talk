"""BEFORE — classical Bayesian inference for the hierarchical Normal-Normal model.

Two methods, both pure ``numpy + scipy`` (PyMC is *not* required):

1. ``conjugate_posterior_known_hyper`` — closed-form per-station Normal-Normal
   conjugate update at *known* ``(mu, tau, sigma)``. Exact, instant, but assumes
   you already know the hyperparameters (rare in practice — the AFTER method does
   not need to assume this).

2. ``mh_sampler`` — a from-scratch Metropolis-Hastings sampler that targets the
   *full* hierarchical posterior
   ``p(theta_1..S, mu, tau, sigma | y)`` with weakly-informative priors and a
   block-wise random-walk proposal. This is what an environmental statistician
   actually does when the hyperparameters are unknown (or PyMC is unavailable).

The model:
    theta_s ~ Normal(mu, tau^2)         partial-pooling prior
    y_{s,i} ~ Normal(theta_s, sigma^2)  observation noise
with weak hyperpriors mu ~ Normal(0, 10^2), tau ~ HalfNormal(5),
sigma ~ HalfNormal(5).

Both methods report per-station posterior means + 95% central credible intervals;
the run harness then evaluates **empirical coverage** of the *true* per-station
``theta_s`` (available only because the data are synthetic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "station_sufficient_stats",
    "conjugate_posterior_known_hyper",
    "mh_sampler",
    "summarize_mh",
]


# --------------------------------------------------------------------------- #
def station_sufficient_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce long-form observations to per-station sufficient statistics.

    For a Normal likelihood the per-station ``(n, mean, var)`` are sufficient —
    every downstream method only needs these, not the raw rows. Returns a
    DataFrame indexed by ``station`` with columns ``['n', 'ybar', 'svar']``
    (``svar`` is the unbiased sample variance, or 0 when ``n == 1``).
    """
    g = df.groupby("station")["value"]
    out = pd.DataFrame({
        "n": g.size().astype(int),
        "ybar": g.mean().astype(float),
        "svar": g.var(ddof=1).fillna(0.0).astype(float),
    })
    return out.sort_index()


# --------------------------------------------------------------------------- #
def conjugate_posterior_known_hyper(
    suff: pd.DataFrame,
    mu: float,
    tau: float,
    sigma: float,
) -> pd.DataFrame:
    """Closed-form per-station posterior ``theta_s | y, mu, tau, sigma``.

    Normal-Normal conjugate update with KNOWN hyperparameters. For each station
    with ``n_s`` observations and sample mean ``ybar_s``::

        posterior_precision = 1/tau^2 + n_s/sigma^2
        posterior_var       = 1 / posterior_precision
        posterior_mean      = posterior_var * (mu/tau^2 + n_s * ybar_s / sigma^2)

    Returns a DataFrame indexed by station with columns
    ``['post_mean', 'post_sd', 'lo95', 'hi95']``.
    """
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be positive")
    n = suff["n"].to_numpy(dtype=float)
    ybar = suff["ybar"].to_numpy(dtype=float)

    prec = 1.0 / (tau * tau) + n / (sigma * sigma)
    var = 1.0 / prec
    mean = var * (mu / (tau * tau) + n * ybar / (sigma * sigma))
    sd = np.sqrt(var)

    z95 = 1.959963984540054   # qnorm(0.975)
    out = pd.DataFrame({
        "post_mean": mean,
        "post_sd": sd,
        "lo95": mean - z95 * sd,
        "hi95": mean + z95 * sd,
    }, index=suff.index)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class MHResult:
    theta: np.ndarray         # (n_keep, S)
    mu: np.ndarray            # (n_keep,)
    tau: np.ndarray           # (n_keep,)
    sigma: np.ndarray         # (n_keep,)
    accept_rate: dict         # per-block acceptance rate
    n_iter: int
    n_burn: int
    n_keep: int


def _log_post(
    theta: np.ndarray, mu: float, tau: float, sigma: float,
    n: np.ndarray, ybar: np.ndarray, ssq: np.ndarray,
    # weak hyperpriors:
    mu_prior_sd: float = 10.0,
    tau_prior_sd: float = 5.0,
    sigma_prior_sd: float = 5.0,
) -> float:
    """Joint log posterior up to an additive constant.

    Uses per-station sufficient stats ``(n, ybar, ssq)`` where
    ``ssq = sum_i (y_{s,i} - ybar_s)^2`` so the observation log-likelihood is
    closed-form: ``-n/2 log(2 pi sigma^2) - [ssq + n (ybar - theta)^2] / (2 sigma^2)``.

    Hyperpriors: ``mu ~ N(0, mu_prior_sd^2)``, ``tau, sigma ~ HalfNormal(*)``
    (so ``log p(tau) = -tau^2 / (2 prior_sd^2)`` for tau > 0, else -inf).
    """
    if tau <= 0 or sigma <= 0:
        return -np.inf
    # Likelihood per station, vectorized over S
    s2 = sigma * sigma
    t2 = tau * tau
    ll_obs = -0.5 * (
        n * np.log(2 * np.pi * s2) + (ssq + n * (ybar - theta) ** 2) / s2
    ).sum()
    # Hierarchical prior on theta
    ll_theta = -0.5 * (
        theta.size * np.log(2 * np.pi * t2) + ((theta - mu) ** 2).sum() / t2
    )
    # Weak hyperpriors (constants dropped)
    lp_mu = -0.5 * (mu / mu_prior_sd) ** 2
    lp_tau = -0.5 * (tau / tau_prior_sd) ** 2
    lp_sigma = -0.5 * (sigma / sigma_prior_sd) ** 2
    return float(ll_obs + ll_theta + lp_mu + lp_tau + lp_sigma)


def mh_sampler(
    suff: pd.DataFrame,
    n_iter: int = 4000,
    n_burn: int = 1500,
    seed: int = 0,
    init: Optional[dict] = None,
    prop_sd_theta: float = 0.35,
    prop_sd_mu: float = 0.25,
    prop_sd_log_tau: float = 0.25,
    prop_sd_log_sigma: float = 0.15,
) -> MHResult:
    """Block-wise random-walk Metropolis-Hastings on (theta, mu, tau, sigma).

    Blocks (each updated per iteration):
      * ``theta`` — coordinate-wise RW Normal proposal (S univariate updates).
      * ``mu``    — RW Normal proposal.
      * ``log tau``, ``log sigma`` — RW Normal proposal in log space
        (the log-Jacobian is +log(x_new) - log(x_old) -> add ``log(new/old)``
        to the acceptance ratio).

    The targets are the *log* joint posterior in :func:`_log_post`. Sufficient
    statistics are pre-computed so each likelihood evaluation is O(S), not O(N).
    """
    if n_iter <= n_burn:
        raise ValueError("n_iter must exceed n_burn")
    rng = np.random.default_rng(seed)

    n_arr = suff["n"].to_numpy(dtype=float)
    ybar = suff["ybar"].to_numpy(dtype=float)
    svar = suff["svar"].to_numpy(dtype=float)
    # ssq = (n-1) * svar ; for n=1 svar=0 -> ssq=0 (consistent)
    ssq = np.maximum(n_arr - 1.0, 0.0) * svar

    S = len(n_arr)

    # ---- initialize at sensible values (pooled mean + sample SDs) ----------- #
    if init is None:
        mu = float(ybar.mean())
        tau = float(max(0.1, np.std(ybar, ddof=1) if S > 1 else 1.0))
        sigma = float(max(0.1, np.sqrt(svar[n_arr > 1].mean()) if (n_arr > 1).any() else 1.0))
        theta = ybar.copy()
    else:
        mu = float(init["mu"]); tau = float(init["tau"]); sigma = float(init["sigma"])
        theta = np.asarray(init["theta"], dtype=float).copy()

    cur_lp = _log_post(theta, mu, tau, sigma, n_arr, ybar, ssq)

    # ---- storage ----------------------------------------------------------- #
    n_keep = n_iter - n_burn
    theta_store = np.empty((n_keep, S))
    mu_store = np.empty(n_keep)
    tau_store = np.empty(n_keep)
    sigma_store = np.empty(n_keep)
    acc = {"theta": 0, "mu": 0, "tau": 0, "sigma": 0}
    tot = {"theta": 0, "mu": 0, "tau": 0, "sigma": 0}

    for it in range(n_iter):
        # ------ theta block: coordinate-wise --------------------------------- #
        for s in range(S):
            old_th = theta[s]
            new_th = old_th + rng.normal(0.0, prop_sd_theta)
            theta[s] = new_th
            new_lp = _log_post(theta, mu, tau, sigma, n_arr, ybar, ssq)
            tot["theta"] += 1
            if np.log(rng.uniform()) < (new_lp - cur_lp):
                cur_lp = new_lp
                acc["theta"] += 1
            else:
                theta[s] = old_th

        # ------ mu block ----------------------------------------------------- #
        new_mu = mu + rng.normal(0.0, prop_sd_mu)
        new_lp = _log_post(theta, new_mu, tau, sigma, n_arr, ybar, ssq)
        tot["mu"] += 1
        if np.log(rng.uniform()) < (new_lp - cur_lp):
            mu = new_mu; cur_lp = new_lp
            acc["mu"] += 1

        # ------ log tau block (Jacobian +log(new/old)) ----------------------- #
        new_tau = tau * np.exp(rng.normal(0.0, prop_sd_log_tau))
        new_lp = _log_post(theta, mu, new_tau, sigma, n_arr, ybar, ssq)
        tot["tau"] += 1
        if np.log(rng.uniform()) < (new_lp - cur_lp + np.log(new_tau) - np.log(tau)):
            tau = new_tau; cur_lp = new_lp
            acc["tau"] += 1

        # ------ log sigma block --------------------------------------------- #
        new_sigma = sigma * np.exp(rng.normal(0.0, prop_sd_log_sigma))
        new_lp = _log_post(theta, mu, tau, new_sigma, n_arr, ybar, ssq)
        tot["sigma"] += 1
        if np.log(rng.uniform()) < (new_lp - cur_lp + np.log(new_sigma) - np.log(sigma)):
            sigma = new_sigma; cur_lp = new_lp
            acc["sigma"] += 1

        if it >= n_burn:
            j = it - n_burn
            theta_store[j] = theta
            mu_store[j] = mu
            tau_store[j] = tau
            sigma_store[j] = sigma

    accept_rate = {k: (acc[k] / tot[k] if tot[k] else 0.0) for k in acc}
    return MHResult(
        theta=theta_store, mu=mu_store, tau=tau_store, sigma=sigma_store,
        accept_rate=accept_rate, n_iter=n_iter, n_burn=n_burn, n_keep=n_keep,
    )


def summarize_mh(mh: MHResult, station_index: Optional[pd.Index] = None) -> pd.DataFrame:
    """Per-station posterior summary (mean + 95% central CI) from MH draws."""
    mean = mh.theta.mean(axis=0)
    lo = np.quantile(mh.theta, 0.025, axis=0)
    hi = np.quantile(mh.theta, 0.975, axis=0)
    sd = mh.theta.std(axis=0, ddof=1)
    idx = station_index if station_index is not None else pd.RangeIndex(mean.size)
    return pd.DataFrame(
        {"post_mean": mean, "post_sd": sd, "lo95": lo, "hi95": hi},
        index=idx,
    )
