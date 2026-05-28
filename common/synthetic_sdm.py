"""Synthetic species distribution-modelling (SDM) dataset.

A reproducible, offline-friendly stand-in for a presence / pseudo-absence (PA)
species-distribution problem over a 2-D geographic grid with three environmental
covariates: **temperature** (T, deg C), **precipitation** (P, mm/yr-like) and
**elevation** (E, standardized). The "true" probability that a site is suitable
for the species is a deliberately **nonlinear function of climate** — a Gaussian
niche in (T, P) whose width is modulated weakly by elevation — so that

  * a **linear / quadratic GLM** (logistic regression with a couple of squared
    terms, the classical SDM baseline) **underfits** the niche surface and
  * a **gradient-boosting** or small-NN model that recovers the unsmooth
    interaction structure **can do better** on AUC / Brier / suitability fit.

The signal is genuine (not noise-driven): if you fit the GLM with a sufficiently
rich basis (centred T, centred P, T^2, P^2, T*P, T^2*E, ...) it eventually
catches up — what the GLM *cannot* discover is the right basis without manual
crafting, which is the BEFORE-vs-AFTER point of Experiment 10.

Outputs (everything in one container ``SDMDataset``):

* a regular 2-D grid (``grid_x``, ``grid_y``, ``grid_shape``) with the three
  covariates and the true suitability surface (noise-free) for evaluation;
* a flat **presence** DataFrame (``presence_df``) sampled from the suitability
  surface with intensity proportional to suitability (an inhomogeneous Poisson
  thinning);
* a flat **background** / pseudo-absence DataFrame (``background_df``) sampled
  uniformly from the same domain (the standard MaxEnt-style background that GLMs
  and ML SDMs both consume);
* the combined ``records_df`` with a binary ``y`` column (1 = presence, 0 =
  background) plus the covariates, suitable as a model input table.

Swap in real data
-----------------
The schema is deliberately simple so it drops in to a real run. See
``REAL_DATA_NOTE`` for an iNaturalist/GBIF + ERA5/WorldClim recipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd

__all__ = [
    "SDMParams",
    "SDMDataset",
    "synthetic_sdm_dataset",
    "REAL_DATA_NOTE",
]

REAL_DATA_NOTE = (
    "To use real data instead of this synthetic field, swap in a presence table "
    "(e.g. GBIF / iNaturalist occurrences for a focal species) and a background "
    "/ pseudo-absence sample from the study area, plus environmental covariates "
    "from a reanalysis (ERA5 2m temperature, total precipitation) and a DEM "
    "(SRTM / Copernicus DEM) or WorldClim bioclimatic layers. Pipeline: "
    "(1) build presence_df with columns [x, y, temperature, precipitation, "
    "elevation, y=1]; (2) sample background_df uniformly (or with a target-group "
    "background) from the same study area with y=0; (3) concatenate into "
    "records_df. Keep these column names and the BEFORE/AFTER scripts run "
    "unchanged. A held-out evaluation grid is optional but recommended for "
    "computing suitability-surface correlation."
)


# --------------------------------------------------------------------------- #
@dataclass
class SDMParams:
    """Knobs of the latent niche surface. Defaults give a regime where a plain
    linear+squared-terms GLM underfits and a GBM / NN clearly catches the niche."""
    # domain (arbitrary projected units, think 100 km square)
    domain: float = 100.0
    grid_res: int = 60

    # climate trends across the domain
    temp_mean_c: float = 16.0
    temp_amp_c: float = 10.0        # warm-to-cold spread across the domain
    precip_mean: float = 900.0      # mm-ish
    precip_amp: float = 700.0
    elev_amp: float = 1.2           # standardized elevation amplitude

    # niche optimum and width (Gaussian in T+P, with elevation interaction)
    niche_t_opt: float = 18.0
    niche_t_sd: float = 3.2
    niche_p_opt: float = 1100.0
    niche_p_sd: float = 280.0
    # interactions that a plain linear+squared-terms GLM CANNOT represent without
    # additional manual crafting:
    #   - the P-axis width depends on elevation (so the niche is a *cross-product*
    #     of squared deviations and elevation, not a simple quadratic in (T,P,E));
    #   - the T-axis optimum *shifts* with elevation (a lapse-rate analogue);
    #   - elevation has its own non-monotone (Gaussian) tolerance.
    p_sd_elev_coef: float = -0.35   # negative => narrower P niche at high elev
    t_opt_elev_coef: float = -2.2   # T optimum drops by 2.2 deg per unit elev
    elev_opt: float = 0.0           # preferred elevation
    elev_sd: float = 1.1            # tolerance around the preferred elevation
    # baseline suitability cap so even the optimum isn't a probability of 1
    max_suitability: float = 0.90

    # observation model
    presence_thinning: float = 1.0  # multiplies suitability before Bernoulli accept
    noise_sd_obs: float = 0.0       # measurement noise on covariates (0 by default)


@dataclass
class SDMDataset:
    """Container for one synthetic SDM problem."""
    # grid / truth
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_shape: Tuple[int, int]
    temperature_field: np.ndarray   # (ny, nx)
    precipitation_field: np.ndarray  # (ny, nx)
    elevation_field: np.ndarray     # (ny, nx)
    suitability_field: np.ndarray   # (ny, nx) -- the true P(suitable | x,y)
    grid_df: pd.DataFrame           # flat (n_grid, [x,y,temperature,precipitation,elevation,suitability])

    # samples
    presence_df: pd.DataFrame       # y=1 rows
    background_df: pd.DataFrame     # y=0 rows
    records_df: pd.DataFrame        # concat of the above (shuffled, reset_index)

    params: SDMParams = field(default_factory=SDMParams)


# --------------------------------------------------------------------------- #
# Latent fields
# --------------------------------------------------------------------------- #
def _temperature_surface(GX: np.ndarray, GY: np.ndarray, p: SDMParams) -> np.ndarray:
    """Smooth N->S cooling + mild E->W tilt; deterministic, no noise."""
    d = p.domain
    yn = GY / d
    xn = GX / d
    return p.temp_mean_c + p.temp_amp_c * (0.5 - yn) + 0.6 * (xn - 0.5)


def _precipitation_surface(GX: np.ndarray, GY: np.ndarray, p: SDMParams,
                           rng: np.random.Generator) -> np.ndarray:
    """Two-scale precipitation pattern (broad gradient + a mid-scale wave).

    Deterministic given ``rng`` (which is seeded from the master seed). No noise
    added; this is a "true climate" field.
    """
    d = p.domain
    xn = GX / d
    yn = GY / d
    broad = p.precip_amp * (0.4 * xn + 0.6 * (1.0 - yn))
    theta = rng.uniform(0, 2 * np.pi)
    k = 2 * np.pi / (0.45 * d)
    wave = 0.35 * p.precip_amp * np.sin(k * (np.cos(theta) * GX + np.sin(theta) * GY))
    return p.precip_mean + broad + wave - 0.5 * p.precip_amp


def _elevation_surface(GX: np.ndarray, GY: np.ndarray, p: SDMParams,
                       rng: np.random.Generator) -> np.ndarray:
    """Multi-scale sinusoidal terrain (standardized to ~unit amplitude * elev_amp)."""
    z = np.zeros_like(GX, dtype=float)
    for (wl, w) in ((55.0, 1.0), (22.0, 0.6), (9.0, 0.35)):
        k = 2 * np.pi / wl
        for _ in range(2):
            theta = rng.uniform(0, 2 * np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            kx, ky = k * np.cos(theta), k * np.sin(theta)
            z = z + w * np.sin(kx * GX + ky * GY + phase)
    # standardize then scale
    z = (z - z.mean()) / (z.std() + 1e-12)
    return p.elev_amp * z


def _suitability_surface(T: np.ndarray, P: np.ndarray, E: np.ndarray,
                         p: SDMParams) -> np.ndarray:
    """Gaussian niche in (T, P, E) with elevation-driven interactions.

    suitability(T, P, E) = max_suit * exp( -0.5 * [ (T - T*(E))^2 / sT^2
                                                    + (P - P*)^2 / sP(E)^2
                                                    + (E - E*)^2 / sE^2 ] )

    where the T optimum shifts with elevation (``T*(E) = T_opt + t_opt_elev_coef
    * E``, a lapse-rate analogue) and the P-axis width also varies with elevation
    (``sP(E) = sP * (1 + p_sd_elev_coef * E)``, clipped to a small positive
    floor). The result is nonlinear and *non-additive* — recovering it inside a
    GLM requires hand-crafted ``T * E``, ``T^2 * E``, ``P^2 * E`` interaction
    terms with the right signs and scales; with only linear + plain squared
    terms in (T, P, E), the GLM structurally underfits.
    """
    sP = p.niche_p_sd * np.clip(1.0 + p.p_sd_elev_coef * E, 0.35, 2.0)
    t_opt = p.niche_t_opt + p.t_opt_elev_coef * E
    dt = (T - t_opt) / p.niche_t_sd
    dp = (P - p.niche_p_opt) / sP
    de = (E - p.elev_opt) / p.elev_sd
    val = p.max_suitability * np.exp(-0.5 * (dt ** 2 + dp ** 2 + de ** 2))
    return val


# --------------------------------------------------------------------------- #
# Public generator
# --------------------------------------------------------------------------- #
def synthetic_sdm_dataset(
    n_pres: int = 400,
    n_bg: int = 2000,
    seed: int = 0,
    params: SDMParams | None = None,
) -> SDMDataset:
    """Generate a synthetic SDM problem (presence + background + truth grid).

    Parameters
    ----------
    n_pres : approximate number of presence (y=1) records to draw. The actual
             count is close to this but not exact, because presences are drawn
             by rejection sampling proportional to the suitability surface.
    n_bg   : number of background (y=0) records, sampled uniformly over the
             domain (the classical MaxEnt-style background).
    seed   : master seed; everything below is deterministic given it.
    params : :class:`SDMParams` (defaults give a regime where a plain GLM
             underfits and a GBM/NN catches the niche).

    Returns
    -------
    :class:`SDMDataset` with the truth grid (suitability surface + covariates),
    presence and background DataFrames, and a combined ``records_df`` with the
    binary ``y`` column.

    Notes
    -----
    Presence sampling uses *inhomogeneous Poisson thinning*: candidate locations
    are drawn uniformly from the domain, evaluated for suitability, and accepted
    with probability ``presence_thinning * suitability``. Iterates until ``n_pres``
    samples are accepted (or until a large pool is exhausted, in which case it
    returns whatever was accepted).
    """
    p = params or SDMParams()
    rng = np.random.default_rng(seed)

    d = p.domain
    res = p.grid_res
    gx = np.linspace(0.0, d, res)
    gy = np.linspace(0.0, d, res)
    GX, GY = np.meshgrid(gx, gy)  # (ny, nx)

    # --- covariates over the grid -------------------------------------------- #
    T_grid = _temperature_surface(GX, GY, p)
    P_grid = _precipitation_surface(GX, GY, p, np.random.default_rng(seed + 11))
    E_grid = _elevation_surface(GX, GY, p, np.random.default_rng(seed + 23))

    suit_grid = _suitability_surface(T_grid, P_grid, E_grid, p)

    # --- flat grid DataFrame ------------------------------------------------- #
    grid_df = pd.DataFrame({
        "x": GX.ravel(),
        "y": GY.ravel(),
        "temperature": T_grid.ravel(),
        "precipitation": P_grid.ravel(),
        "elevation": E_grid.ravel(),
        "suitability": suit_grid.ravel(),
    })

    # --- sample background (uniform over the domain) ------------------------- #
    bg_rng = np.random.default_rng(seed + 37)
    bg_x = bg_rng.uniform(0.0, d, n_bg)
    bg_y = bg_rng.uniform(0.0, d, n_bg)
    bg_T = _interp_field(gx, gy, T_grid, bg_x, bg_y)
    bg_P = _interp_field(gx, gy, P_grid, bg_x, bg_y)
    bg_E = _interp_field(gx, gy, E_grid, bg_x, bg_y)
    background_df = pd.DataFrame({
        "x": bg_x, "y": bg_y,
        "temperature": bg_T, "precipitation": bg_P, "elevation": bg_E,
        "y_label": np.zeros(n_bg, dtype=int),
    })

    # --- sample presences via inhomogeneous Poisson thinning ----------------- #
    pr_rng = np.random.default_rng(seed + 53)
    # cap the candidate pool so the loop is always bounded
    pool_size = int(max(20 * n_pres, 5000))
    accepted_x, accepted_y, accepted_T, accepted_P, accepted_E = [], [], [], [], []
    drawn = 0
    while len(accepted_x) < n_pres and drawn < pool_size:
        batch = min(2 * (n_pres - len(accepted_x)) + 200, pool_size - drawn)
        cx = pr_rng.uniform(0.0, d, batch)
        cy = pr_rng.uniform(0.0, d, batch)
        cT = _interp_field(gx, gy, T_grid, cx, cy)
        cP = _interp_field(gx, gy, P_grid, cx, cy)
        cE = _interp_field(gx, gy, E_grid, cx, cy)
        suit = _suitability_surface(cT, cP, cE, p)
        prob = np.clip(p.presence_thinning * suit, 0.0, 1.0)
        u = pr_rng.random(batch)
        keep = u < prob
        accepted_x.extend(cx[keep].tolist())
        accepted_y.extend(cy[keep].tolist())
        accepted_T.extend(cT[keep].tolist())
        accepted_P.extend(cP[keep].tolist())
        accepted_E.extend(cE[keep].tolist())
        drawn += batch
        # trim if we overshot slightly
        if len(accepted_x) > n_pres:
            accepted_x = accepted_x[:n_pres]
            accepted_y = accepted_y[:n_pres]
            accepted_T = accepted_T[:n_pres]
            accepted_P = accepted_P[:n_pres]
            accepted_E = accepted_E[:n_pres]
            break

    presence_df = pd.DataFrame({
        "x": np.asarray(accepted_x, dtype=float),
        "y": np.asarray(accepted_y, dtype=float),
        "temperature": np.asarray(accepted_T, dtype=float),
        "precipitation": np.asarray(accepted_P, dtype=float),
        "elevation": np.asarray(accepted_E, dtype=float),
        "y_label": np.ones(len(accepted_x), dtype=int),
    })

    # --- combined records (shuffled deterministically) ----------------------- #
    records_df = pd.concat([presence_df, background_df], axis=0, ignore_index=True)
    shuffle_rng = np.random.default_rng(seed + 71)
    idx = np.arange(len(records_df))
    shuffle_rng.shuffle(idx)
    records_df = records_df.iloc[idx].reset_index(drop=True)

    return SDMDataset(
        grid_x=gx,
        grid_y=gy,
        grid_shape=(res, res),
        temperature_field=T_grid,
        precipitation_field=P_grid,
        elevation_field=E_grid,
        suitability_field=suit_grid,
        grid_df=grid_df,
        presence_df=presence_df,
        background_df=background_df,
        records_df=records_df,
        params=p,
    )


# --------------------------------------------------------------------------- #
def _interp_field(gx: np.ndarray, gy: np.ndarray, field2d: np.ndarray,
                  xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a regular-grid field at scattered (x,y) points.

    Inputs:
        gx, gy   : strictly increasing 1-D axes of length nx, ny
        field2d  : (ny, nx) values on the grid
        xs, ys   : (m,) query coordinates inside [gx[0], gx[-1]] x [gy[0], gy[-1]]
    """
    nx = len(gx)
    ny = len(gy)
    # locate cells
    ix = np.clip(np.searchsorted(gx, xs, side="right") - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(gy, ys, side="right") - 1, 0, ny - 2)
    x0 = gx[ix]; x1 = gx[ix + 1]
    y0 = gy[iy]; y1 = gy[iy + 1]
    tx = np.clip((xs - x0) / (x1 - x0), 0.0, 1.0)
    ty = np.clip((ys - y0) / (y1 - y0), 0.0, 1.0)
    f00 = field2d[iy, ix]
    f10 = field2d[iy, ix + 1]
    f01 = field2d[iy + 1, ix]
    f11 = field2d[iy + 1, ix + 1]
    return (f00 * (1 - tx) * (1 - ty)
            + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty
            + f11 * tx * ty)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    ds = synthetic_sdm_dataset(n_pres=300, n_bg=1500, seed=0)
    print("grid:", ds.grid_shape,
          "T range:", round(float(ds.temperature_field.min()), 2),
          round(float(ds.temperature_field.max()), 2))
    print("P range:", round(float(ds.precipitation_field.min()), 1),
          round(float(ds.precipitation_field.max()), 1))
    print("suitability range:", round(float(ds.suitability_field.min()), 3),
          round(float(ds.suitability_field.max()), 3),
          "mean:", round(float(ds.suitability_field.mean()), 3))
    print("presence n:", len(ds.presence_df), "background n:", len(ds.background_df))
    print("records y mean (prevalence):",
          round(float(ds.records_df["y_label"].mean()), 3))
