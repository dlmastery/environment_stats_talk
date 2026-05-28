"""AFTER — amortized Bayesian inference for the hierarchical Normal-Normal model.

Idea (the "amortized inference" / "simulation-based inference" pattern):
instead of running MCMC every time a new dataset arrives, we **train once** on
many simulated datasets from the *same* model family and learn a fast neural
mapping from observed summary statistics to the posterior parameters of each
station's latent effect ``theta_s``. After training, scoring a fresh dataset is
a single CPU forward pass (microseconds per station) regardless of dataset size
— while the classical MCMC pays the full per-dataset cost every time.

Network input (per station, dataset-aware):
    [n_s, ybar_s, ybar_s - global_ybar, log(1+n_s),
     global_ybar, log(sd_across_ybars + 1e-6), log(sd_within_pooled + 1e-6),
     S (count of stations in this dataset, log-scaled)]

Network output (per station):
    [mean_hat, log_sd_hat]  -> posterior_mean = mean_hat,
                               posterior_sd  = exp(log_sd_hat)

Training loss: **Gaussian negative log-likelihood** of the true ``theta_s`` under
``N(mean_hat, exp(log_sd_hat)^2)`` — a proper scoring rule, so the network is
*encouraged* to be calibrated (sharp + honest), not just accurate. Generative
hyperparameters of each simulated dataset are randomized within a realistic
range so the trained network generalizes across plausible regimes (this is the
"amortized over a family of priors" trick).

Pure PyTorch (CPU is fine; auto-uses CUDA if present). Deterministic seeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

# Import the synthetic generator from its submodule directly (not common/__init__.py)
from common.synthetic_bayesian import HierarchyParams, synthetic_station_offsets

__all__ = [
    "AmortizedConfig",
    "AmortizedPosterior",
    "build_features_from_suff",
    "simulate_training_batch",
    "train_amortized",
    "score_amortized",
]


# --------------------------------------------------------------------------- #
@dataclass
class AmortizedConfig:
    """Knobs for the amortized network and its training schedule."""

    hidden: int = 64
    n_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_train_datasets: int = 4000     # outer simulation budget (datasets)
    batch_datasets: int = 64         # datasets per gradient step
    n_epochs: int = 1                # passes over the simulation budget
    # range over which hyperparameters of each training dataset are drawn
    mu_range: Tuple[float, float] = (-2.0, 2.0)
    tau_range: Tuple[float, float] = (0.3, 2.5)
    sigma_range: Tuple[float, float] = (0.3, 2.0)
    s_min: int = 8
    s_max: int = 40
    n_per_station_min: int = 4
    n_per_station_max: int = 24
    seed: int = 0


# --------------------------------------------------------------------------- #
class AmortizedPosterior(nn.Module):
    """Small MLP mapping per-station features -> (mean, log_sd) for theta_s."""

    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2)   # (mean, log_sd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        out = self.head(h)
        # Clamp log_sd for numerical stability: sigma in [exp(-5), exp(3)] ~ [0.007, 20]
        out = torch.cat([out[..., :1], out[..., 1:].clamp(-5.0, 3.0)], dim=-1)
        return out


# --------------------------------------------------------------------------- #
N_FEATURES = 8


def build_features_from_suff(suff: pd.DataFrame) -> np.ndarray:
    """Vectorized per-station features for a single dataset.

    Returns an ``(S, N_FEATURES)`` numpy array. The features are deliberately
    cheap to compute and capture both per-station signal (``n_s``, ``ybar_s``)
    and dataset-level context (global mean, between/within spread, S) so the
    network can learn the *shrinkage* trade-off.
    """
    n = suff["n"].to_numpy(dtype=float)
    ybar = suff["ybar"].to_numpy(dtype=float)
    svar = suff["svar"].to_numpy(dtype=float)
    S = len(n)
    global_ybar = float(ybar.mean()) if S > 0 else 0.0
    sd_between = float(ybar.std(ddof=1)) if S > 1 else 0.0
    # pooled within-station sd, ignoring n=1
    mask = n > 1
    if mask.any():
        sd_within = float(np.sqrt((svar[mask] * (n[mask] - 1)).sum()
                                  / max(1.0, (n[mask] - 1).sum())))
    else:
        sd_within = 0.0
    log_S = float(np.log1p(S))

    feats = np.column_stack([
        n,
        ybar,
        ybar - global_ybar,
        np.log1p(n),
        np.full(S, global_ybar),
        np.full(S, np.log(sd_between + 1e-6)),
        np.full(S, np.log(sd_within + 1e-6)),
        np.full(S, log_S),
    ])
    return feats.astype(np.float32)


# --------------------------------------------------------------------------- #
def _sample_one_dataset(rng: np.random.Generator, cfg: AmortizedConfig):
    """Draw one synthetic dataset from the randomized generative prior."""
    mu = float(rng.uniform(*cfg.mu_range))
    tau = float(rng.uniform(*cfg.tau_range))
    sigma = float(rng.uniform(*cfg.sigma_range))
    S = int(rng.integers(cfg.s_min, cfg.s_max + 1))
    params = HierarchyParams(
        mu=mu, tau=tau, sigma=sigma,
        n_per_station_min=cfg.n_per_station_min,
        n_per_station_max=cfg.n_per_station_max,
    )
    # Use a fresh seed derived from rng so each sim is deterministic but distinct.
    sub_seed = int(rng.integers(0, 2**31 - 1))
    df, truth = synthetic_station_offsets(n_stations=S, seed=sub_seed, params=params)
    return df, truth


def simulate_training_batch(
    cfg: AmortizedConfig, n_datasets: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (features, targets) by simulating ``n_datasets`` Bayesian datasets.

    Returns:
        X : (sum_S, N_FEATURES)  per-station feature rows across all datasets.
        y : (sum_S,)             ground-truth ``theta_s`` for each row.
    """
    # local import to keep this file's surface clean
    from before.classical_bayes import station_sufficient_stats  # noqa: WPS433

    xs, ys = [], []
    for _ in range(n_datasets):
        df, truth = _sample_one_dataset(rng, cfg)
        suff = station_sufficient_stats(df)
        feats = build_features_from_suff(suff)
        xs.append(feats)
        ys.append(truth["theta"].astype(np.float32))
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


# --------------------------------------------------------------------------- #
def _gauss_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Proper-scoring negative log-likelihood under N(mean, sigma^2)."""
    mean = out[..., 0]
    log_sd = out[..., 1]
    sd = torch.exp(log_sd)
    # -log N(y; mean, sd) up to a constant (per element)
    return (log_sd + 0.5 * ((y - mean) / sd) ** 2).mean()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_amortized(
    cfg: AmortizedConfig,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """Train the amortized network on simulated datasets. Returns model+stats."""
    device = device or get_device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)

    model = AmortizedPosterior(N_FEATURES, hidden=cfg.hidden, n_layers=cfg.n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_steps = max(1, cfg.n_train_datasets // cfg.batch_datasets)
    losses = []
    # Pre-compute global feature normalizer on a small initial batch.
    X0, _ = simulate_training_batch(cfg, n_datasets=min(64, cfg.batch_datasets), rng=rng)
    feat_mean = X0.mean(axis=0)
    feat_std = X0.std(axis=0) + 1e-6
    feat_mean_t = torch.as_tensor(feat_mean, dtype=torch.float32, device=device)
    feat_std_t = torch.as_tensor(feat_std, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(cfg.n_epochs):
        for step in range(n_steps):
            X, y = simulate_training_batch(cfg, n_datasets=cfg.batch_datasets, rng=rng)
            Xb = (torch.as_tensor(X, device=device) - feat_mean_t) / feat_std_t
            yb = torch.as_tensor(y, device=device)
            opt.zero_grad(set_to_none=True)
            out = model(Xb)
            loss = _gauss_nll(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if verbose and (step % max(1, n_steps // 5) == 0):
                print(f"  [amortized] epoch={epoch} step={step}/{n_steps} loss={losses[-1]:.4f}")

    model.eval()
    stats = {
        "loss_history": losses,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_steps": n_steps * cfg.n_epochs,
        "device": str(device),
    }
    return model, stats


# --------------------------------------------------------------------------- #
def score_amortized(
    model: AmortizedPosterior,
    suff: pd.DataFrame,
    stats: dict,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """One forward pass: per-station posterior mean + 95% CI for a fresh dataset."""
    device = device or get_device()
    feats = build_features_from_suff(suff)
    Xt = (torch.as_tensor(feats, device=device)
          - torch.as_tensor(stats["feat_mean"], device=device)) \
         / torch.as_tensor(stats["feat_std"], device=device)
    with torch.no_grad():
        out = model(Xt).cpu().numpy()
    mean = out[:, 0]
    sd = np.exp(out[:, 1])
    z95 = 1.959963984540054
    return pd.DataFrame({
        "post_mean": mean,
        "post_sd": sd,
        "lo95": mean - z95 * sd,
        "hi95": mean + z95 * sd,
    }, index=suff.index)
