"""Synthetic spatial field generator for the interpolation experiment (Exp 06).

This produces *scattered station observations* of an environmental variable over a
2D domain plus a dense, regular evaluation grid — fully synthetic and deterministic
so the spatial-interpolation experiment is reproducible offline (no network, no
geostatistics-package download). It is built so that

  * **coordinate-only interpolation** (IDW, ordinary kriging / a GP on (x, y)) does
    a respectable job — there is genuine spatial autocorrelation to exploit — but
  * **covariate-aware machine learning** does *better*, because a large part of the
    field's variation is driven by an auxiliary covariate (think **elevation**) that
    a coordinate-only interpolator never sees.

Why this generator exists
--------------------------
The textbook environmental example is mapping **temperature** (or a pollutant) from
a sparse monitoring network. Temperature has two distinct components:

1. A **smooth, spatially-autocorrelated** component — nearby places are similar.
   Classical geostatistics (kriging / variograms) was built to model exactly this
   and is excellent at it.
2. A **covariate-driven** component — temperature falls with **elevation** (a lapse
   rate), and elevation can change sharply over short distances (a valley next to a
   ridge). Two stations a few km apart can therefore have very different values even
   though they are "close" in (x, y). A coordinate-only interpolator must smooth
   across that, leaving real, *explainable* error on the table. A model that is
   *given the elevation* at the prediction location recovers it directly.

The latent surface is therefore

    z(x, y) = trend(x, y) + lapse * elevation(x, y) + grf(x, y)

where ``trend`` is a smooth large-scale gradient in coordinates, ``elevation`` is a
deterministic multi-scale terrain surface (the covariate), ``lapse`` is the (mostly
negative) sensitivity to the covariate, and ``grf`` is a **spatially-correlated
Gaussian random field** (a squared-exponential covariance) standing in for the
"everything else that is locally smooth" residual. Stations sample this surface at
scattered locations with a little measurement noise; the test grid stores the
**noise-free truth** so held-out error is measured against the real field.

Because the GRF is coordinate-smooth, ordinary kriging on (x, y) captures the
``trend + grf`` part well. The ``lapse * elevation`` part is *not* a smooth function
of (x, y) (terrain has short-scale structure), so only a covariate-aware model can
recover it — that is the BEFORE/AFTER gap, and it is real, not rigged by noise.

Swapping in real data
---------------------
``REAL_DATA_NOTE`` documents how to drop in a real station network (e.g. GHCN /
OpenAQ / a national met service) plus a **DEM** (digital elevation model, e.g.
SRTM / Copernicus DEM) as the covariate. Keep the same return schema and every
downstream script (IDW, kriging, ML, runner, tests) works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SpatialFieldParams",
    "SpatialDataset",
    "synthetic_spatial_field",
    "REAL_DATA_NOTE",
]

REAL_DATA_NOTE = (
    "To use real data instead of this synthetic field, swap in a station network "
    "and a covariate raster: e.g. observed 2 m temperature from GHCN-Daily / a "
    "national met service (or a pollutant from OpenAQ), and ELEVATION sampled from "
    "a DEM (SRTM or the Copernicus DEM) at each station and at every prediction "
    "cell. Assemble: (1) train arrays coords[n,2], covariate[n], value[n]; (2) a "
    "dense grid grid_coords[m,2], grid_covariate[m] (DEM resampled to the target "
    "grid) and, for scoring, grid_value[m] from a held-out station subset or a "
    "trusted reanalysis. Keep these names/shapes and IDW, kriging, the covariate-"
    "aware ML, the runner and the tests all work unchanged. (Standardise coords to "
    "a projected CRS in km and scale the covariate as done here.)"
)


@dataclass
class SpatialFieldParams:
    """Knobs controlling the latent surface. Defaults give a field where kriging is
    good but covariate-aware ML is clearly better."""
    domain: float = 100.0           # square domain side length (arbitrary km units)
    trend_amp: float = 4.0          # amplitude of the smooth large-scale (x,y) trend
    grf_amp: float = 3.0            # std of the spatially-correlated residual (GRF)
    grf_length: float = 18.0        # GRF correlation length (km) — the smooth part
    lapse: float = -7.0             # covariate sensitivity (e.g. °C per unit elev)
    elev_amp: float = 1.0           # amplitude of the (standardised) elevation field
    # Multi-scale terrain: list of (wavelength_km, weight). Short wavelengths give
    # the covariate sharp local structure that coordinate-only interpolation cannot
    # follow, which is what makes the covariate genuinely informative.
    terrain_scales: tuple = ((60.0, 1.0), (25.0, 0.7), (9.0, 0.45))
    noise_sd: float = 0.35          # station measurement-noise std (on the value)
    value_offset: float = 15.0      # additive offset so values look like a variable


@dataclass
class SpatialDataset:
    """Container for one synthetic spatial problem.

    Train (scattered stations):
        train_coords     : (n_train, 2) station x,y locations
        train_covariate  : (n_train,)   covariate (e.g. elevation) at each station
        train_value      : (n_train,)   NOISY observed value at each station
        train_value_clean: (n_train,)   noise-free truth at each station (for tests)

    Test (dense regular grid):
        grid_coords      : (n_grid, 2)  grid cell x,y locations (flattened)
        grid_covariate   : (n_grid,)    covariate at each grid cell
        grid_value       : (n_grid,)    NOISE-FREE true field at each grid cell
        grid_shape       : (ny, nx)     so predictions can be reshaped to a map
        grid_x, grid_y   : 1-D axis coordinates (length nx, ny)

    Covariate field (for plotting the DEM-like surface):
        cov_field        : (ny, nx)     covariate over the grid as a 2-D image
    """
    train_coords: np.ndarray
    train_covariate: np.ndarray
    train_value: np.ndarray
    train_value_clean: np.ndarray
    grid_coords: np.ndarray
    grid_covariate: np.ndarray
    grid_value: np.ndarray
    grid_shape: tuple
    grid_x: np.ndarray
    grid_y: np.ndarray
    cov_field: np.ndarray
    params: SpatialFieldParams = field(default_factory=SpatialFieldParams)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _elevation_surface(x: np.ndarray, y: np.ndarray, p: SpatialFieldParams,
                       rng: np.random.Generator) -> np.ndarray:
    """A deterministic multi-scale terrain surface evaluated at points (x, y).

    Built as a sum of sinusoids at several wavelengths with random phases/orientations
    (a cheap fractal-terrain surrogate). Short wavelengths give the covariate sharp
    LOCAL structure — the part a coordinate-only interpolator cannot reproduce.
    Returned standardised to ~zero-mean, unit-amplitude then scaled by ``elev_amp``.
    """
    z = np.zeros_like(x, dtype=float)
    for (wl, w) in p.terrain_scales:
        k = 2 * np.pi / float(wl)
        # two random-orientation plane waves per scale
        for _ in range(2):
            theta = rng.uniform(0, 2 * np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            kx, ky = k * np.cos(theta), k * np.sin(theta)
            z = z + w * np.sin(kx * x + ky * y + phase)
    # standardise (using this evaluation's own moments is fine: the same surface is
    # used for both train and grid via the closed form below, see make).
    z = (z - z.mean()) / (z.std() + 1e-12)
    return p.elev_amp * z


def _smooth_trend(x: np.ndarray, y: np.ndarray, p: SpatialFieldParams) -> np.ndarray:
    """A smooth large-scale coordinate trend (a gentle tilt + broad bump) that
    coordinate-only interpolation CAN capture."""
    d = p.domain
    xn, yn = x / d, y / d
    tilt = p.trend_amp * (0.6 * xn + 0.4 * yn)
    bump = 0.5 * p.trend_amp * np.exp(-(((xn - 0.5) ** 2 + (yn - 0.4) ** 2) / 0.08))
    return tilt + bump


def _grf(coords: np.ndarray, p: SpatialFieldParams,
         rng: np.random.Generator) -> np.ndarray:
    """Draw a zero-mean spatially-correlated Gaussian random field at ``coords``
    using a squared-exponential (Gaussian) covariance with length ``grf_length``.

    Sampled jointly over all requested points (train + grid together) so the field
    is mutually consistent. Uses an eigen/Cholesky factor of the covariance matrix.
    """
    n = coords.shape[0]
    # pairwise squared distances
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)
    cov = (p.grf_amp ** 2) * np.exp(-0.5 * d2 / (p.grf_length ** 2))
    cov = cov + 1e-8 * np.eye(n)  # jitter for numerical PD
    # Cholesky (fall back to eigh if needed).
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 0.0, None)
        L = V @ np.diag(np.sqrt(w))
    g = L @ rng.standard_normal(n)
    return g


# --------------------------------------------------------------------------- #
# Public generator
# --------------------------------------------------------------------------- #
def synthetic_spatial_field(
    n_points: int = 160,
    seed: int = 0,
    grid_res: int = 40,
    params: SpatialFieldParams | None = None,
) -> SpatialDataset:
    """Generate scattered station observations + a dense test grid of an
    environmental variable driven by coordinates AND a covariate (elevation).

    Parameters
    ----------
    n_points : number of scattered training stations.
    seed     : master seed; everything below is deterministic given it.
    grid_res : the dense evaluation grid is ``grid_res x grid_res`` cells.
    params   : :class:`SpatialFieldParams` (defaults give kriging-good,
               covariate-ML-better behaviour).

    Returns
    -------
    :class:`SpatialDataset` — train stations (coords, covariate, value), a dense
    grid (coords, covariate, noise-free truth, shape/axes) and the covariate field.

    Notes
    -----
    The latent surface ``z = trend(x,y) + lapse*elev(x,y) + grf(x,y)`` is sampled
    jointly over (stations ∪ grid) so the spatially-correlated residual is mutually
    consistent. Stations get small measurement noise; the grid stores the noise-free
    truth so held-out error is against the real field.
    """
    p = params or SpatialFieldParams()
    rng = np.random.default_rng(seed)

    d = p.domain
    # --- scattered station locations (uniform over the domain) ------------------ #
    train_coords = rng.uniform(0.0, d, size=(n_points, 2))

    # --- dense regular evaluation grid ----------------------------------------- #
    gx = np.linspace(0.0, d, grid_res)
    gy = np.linspace(0.0, d, grid_res)
    GX, GY = np.meshgrid(gx, gy)               # shape (ny, nx) = (grid_res, grid_res)
    grid_coords = np.column_stack([GX.ravel(), GY.ravel()])

    # --- covariate (elevation) over BOTH stations and grid, from one surface ---- #
    # Use a dedicated sub-generator so the terrain is identical regardless of how
    # many stations are requested (deterministic given seed).
    terr_rng = np.random.default_rng(seed + 11)
    all_xy = np.vstack([train_coords, grid_coords])
    elev_all = _elevation_surface(all_xy[:, 0], all_xy[:, 1], p, terr_rng)
    train_cov = elev_all[:n_points]
    grid_cov = elev_all[n_points:]

    # --- smooth coordinate trend ----------------------------------------------- #
    trend_all = _smooth_trend(all_xy[:, 0], all_xy[:, 1], p)

    # --- spatially-correlated residual (GRF), jointly over all points ----------- #
    grf_rng = np.random.default_rng(seed + 23)
    grf_all = _grf(all_xy, p, grf_rng)

    # --- assemble the latent surface ------------------------------------------- #
    latent_all = p.value_offset + trend_all + p.lapse * elev_all + grf_all
    train_clean = latent_all[:n_points]
    grid_value = latent_all[n_points:]          # noise-free truth on the grid

    # station measurement noise
    obs_rng = np.random.default_rng(seed + 37)
    train_value = train_clean + obs_rng.normal(0.0, p.noise_sd, size=n_points)

    cov_field = grid_cov.reshape(grid_res, grid_res)

    return SpatialDataset(
        train_coords=train_coords,
        train_covariate=train_cov,
        train_value=train_value,
        train_value_clean=train_clean,
        grid_coords=grid_coords,
        grid_covariate=grid_cov,
        grid_value=grid_value,
        grid_shape=(grid_res, grid_res),
        grid_x=gx,
        grid_y=gy,
        cov_field=cov_field,
        params=p,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    ds = synthetic_spatial_field(n_points=160, seed=0, grid_res=40)
    print("train coords  :", ds.train_coords.shape)
    print("train value   :", ds.train_value.shape, "mean", round(float(ds.train_value.mean()), 3))
    print("grid coords   :", ds.grid_coords.shape, "shape", ds.grid_shape)
    print("grid value    : min/max",
          round(float(ds.grid_value.min()), 2), round(float(ds.grid_value.max()), 2))
    # How informative is the covariate? Correlation of value with elevation.
    r = np.corrcoef(ds.grid_value, ds.grid_covariate)[0, 1]
    print("corr(value, covariate) on grid:", round(float(r), 3))
    # Fraction of variance the covariate explains (univariate linear).
    cov = ds.grid_covariate
    val = ds.grid_value
    b = np.polyfit(cov, val, 1)
    pred = np.polyval(b, cov)
    r2 = 1 - np.sum((val - pred) ** 2) / np.sum((val - val.mean()) ** 2)
    print("univariate R^2 of covariate on grid value:", round(float(r2), 3))
