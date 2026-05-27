"""BEFORE — the traditional coordinate-only spatial interpolators.

This is what an environmental statistician / geostatistician reaches for to map a
variable from a sparse monitoring network using **location alone**:

* **Inverse-Distance Weighting (IDW)** — a transparent, parameter-light baseline:
  predict each location as a distance-weighted average of the observations,
  ``ẑ(s₀) = Σ wᵢ zᵢ / Σ wᵢ`` with ``wᵢ = 1 / dᵢᵖ``. No model fitting; just a power.
* **Ordinary kriging via a Gaussian process** — the geostatistical workhorse. We
  use scikit-learn's :class:`GaussianProcessRegressor` with an **RBF (squared-
  exponential) kernel + WhiteKernel** nugget, fit on the station coordinates only.
  A GP with a stationary kernel is the Bayesian twin of ordinary kriging: the kernel
  *is* the (Gaussian) covariance model, its length scale is fitted by maximising the
  marginal likelihood (the modern stand-in for fitting a variogram by eye), and it
  yields a full **predictive variance** surface — the principled uncertainty map that
  is kriging's signature and that vanilla ML does not give you for free.

Both methods see **(x, y) only** — never the covariate. That is the point: they
model spatial autocorrelation well, but cannot recover the part of the field driven
by a short-scale covariate (elevation), which is what the AFTER model exploits.

We also expose :func:`empirical_variogram` so the runner can draw the empirical
semivariogram (the classic "is there spatial structure to exploit?" diagnostic).

Contract (shared with the AFTER side so the orchestrator treats them uniformly):

    run_before(dataset, ...) -> dict with per-method
        {"rmse":.., "mae":.., "y_pred":[...]}  (+ kriging adds "y_std")

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.metrics import rmse, mae  # noqa: E402

__all__ = [
    "idw_predict",
    "fit_kriging",
    "kriging_predict",
    "empirical_variogram",
    "run_before",
]


# --------------------------------------------------------------------------- #
# Inverse-distance weighting
# --------------------------------------------------------------------------- #
def idw_predict(train_coords: np.ndarray, train_value: np.ndarray,
                query_coords: np.ndarray, power: float = 2.0,
                eps: float = 1e-9) -> np.ndarray:
    """Inverse-distance-weighted interpolation (coords only).

    ``ẑ(s₀) = Σ wᵢ zᵢ / Σ wᵢ`` with ``wᵢ = 1 / max(dᵢ, eps)ᵖ``. The ``eps`` floor keeps
    a query that coincides with a station finite (it then collapses to that station's
    value, as IDW should). Vectorised over all query points.
    """
    tc = np.asarray(train_coords, float)
    tv = np.asarray(train_value, float)
    qc = np.asarray(query_coords, float)
    # pairwise distances (n_query, n_train)
    d = np.sqrt(((qc[:, None, :] - tc[None, :, :]) ** 2).sum(axis=-1))
    d = np.maximum(d, eps)
    w = 1.0 / (d ** power)
    return (w @ tv) / w.sum(axis=1)


# --------------------------------------------------------------------------- #
# Ordinary kriging via a Gaussian process (RBF + nugget)
# --------------------------------------------------------------------------- #
def fit_kriging(train_coords: np.ndarray, train_value: np.ndarray,
                length_scale: float = 15.0, n_restarts: int = 3,
                seed: int = 0):
    """Fit an ordinary-kriging-style GP on station coordinates only.

    Kernel = ConstantKernel * RBF(length_scale) + WhiteKernel(nugget). The RBF is the
    Gaussian covariance model of geostatistics; its length scale and the nugget are
    learned by maximising the marginal likelihood (the modern, automatic alternative
    to fitting a variogram by hand). ``normalize_y`` handles the non-zero mean
    (ordinary, not simple, kriging). Returns the fitted regressor.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, WhiteKernel, ConstantKernel,
    )

    kernel = (
        ConstantKernel(10.0, (1e-2, 1e4))
        * RBF(length_scale, (1.0, 100.0))
        + WhiteKernel(0.5, (1e-3, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=int(n_restarts),
        random_state=int(seed),
    )
    gp.fit(np.asarray(train_coords, float), np.asarray(train_value, float))
    return gp


def kriging_predict(gp, query_coords: np.ndarray):
    """Predict mean and standard deviation at the query locations.

    Returns ``(mean, std)``. The ``std`` is the kriging/GP predictive standard
    deviation — the principled uncertainty surface that is the whole reason a
    geostatistician reaches for kriging. (Vanilla RF/GBM give no such calibrated map.)
    """
    mean, std = gp.predict(np.asarray(query_coords, float), return_std=True)
    return mean, std


# --------------------------------------------------------------------------- #
# Empirical (semi)variogram — the classic spatial-structure diagnostic
# --------------------------------------------------------------------------- #
def empirical_variogram(coords: np.ndarray, values: np.ndarray,
                        n_bins: int = 15, max_dist: float | None = None):
    """Compute a classical (Matheron) empirical semivariogram.

    γ(h) = (1 / 2N(h)) Σ (z(sᵢ) − z(sⱼ))²  over all pairs whose separation falls in
    the lag bin centred at ``h``. Returns ``(lag_centres, gamma, counts)``. This is
    the plot a geostatistician stares at to decide there *is* exploitable spatial
    autocorrelation (γ rises with distance toward a sill) before kriging.
    """
    c = np.asarray(coords, float)
    v = np.asarray(values, float)
    n = len(c)
    iu, ju = np.triu_indices(n, k=1)
    d = np.sqrt(((c[iu] - c[ju]) ** 2).sum(axis=1))
    sq = 0.5 * (v[iu] - v[ju]) ** 2
    if max_dist is None:
        max_dist = float(np.percentile(d, 90))
    edges = np.linspace(0.0, max_dist, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    idx = np.digitize(d, edges) - 1
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b] > 0:
            gamma[b] = float(sq[sel].mean())
    return centres, gamma, counts


# --------------------------------------------------------------------------- #
# Orchestration entry point
# --------------------------------------------------------------------------- #
def run_before(dataset, idw_power: float = 2.0, length_scale: float = 15.0,
               n_restarts: int = 3, seed: int = 0) -> dict:
    """Run the BEFORE coordinate-only interpolators on a :class:`SpatialDataset`.

    Output schema:
        {
          "idw":     {"rmse":.., "mae":.., "y_pred":[...]},
          "kriging": {"rmse":.., "mae":.., "y_pred":[...], "y_std":[...],
                      "kernel":"...", "length_scale":..},
        }
    All metrics are computed against the dense grid's noise-free truth.
    """
    tc = dataset.train_coords
    tv = dataset.train_value
    qc = dataset.grid_coords
    yte = dataset.grid_value

    out: dict = {}

    # IDW
    yp_idw = idw_predict(tc, tv, qc, power=idw_power)
    out["idw"] = {
        "rmse": rmse(yte, yp_idw),
        "mae": mae(yte, yp_idw),
        "y_pred": np.asarray(yp_idw, float).tolist(),
        "power": float(idw_power),
    }

    # Ordinary kriging (GP)
    gp = fit_kriging(tc, tv, length_scale=length_scale,
                     n_restarts=n_restarts, seed=seed)
    yp_k, ystd_k = kriging_predict(gp, qc)
    # recover the fitted RBF length scale for reporting
    try:
        fitted_ls = float(gp.kernel_.k1.k2.length_scale)
    except Exception:  # pragma: no cover - kernel structure fallback
        fitted_ls = float("nan")
    out["kriging"] = {
        "rmse": rmse(yte, yp_k),
        "mae": mae(yte, yp_k),
        "y_pred": np.asarray(yp_k, float).tolist(),
        "y_std": np.asarray(ystd_k, float).tolist(),
        "kernel": str(gp.kernel_),
        "length_scale": fitted_ls,
        "mean_pred_std": float(np.mean(ystd_k)),
    }
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_spatial import synthetic_spatial_field

    ds = synthetic_spatial_field(n_points=160, seed=0, grid_res=30)
    res = run_before(ds)
    for name, m in res.items():
        print(f"{name:8s} rmse={m['rmse']:.3f} mae={m['mae']:.3f}")
    centres, gamma, counts = empirical_variogram(ds.train_coords, ds.train_value)
    print("variogram lags:", np.round(centres, 1))
    print("variogram gamma:", np.round(gamma, 2))
