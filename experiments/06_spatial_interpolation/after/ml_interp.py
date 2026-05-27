"""AFTER — covariate-aware machine-learning spatial interpolation.

Where the BEFORE interpolators see **(x, y) only**, the AFTER model is *given the
covariate* (elevation) at every prediction location, plus a few cheap engineered
features. That lets it recover the covariate-driven part of the field — the lapse
effect that varies on short spatial scales and that a coordinate-only interpolator
must smooth across.

Model
-----
A scikit-learn tree ensemble (**RandomForest** by default; **GradientBoosting**
selectable) trained on features

    [x, y, covariate, x·y, covariate², x·covariate, y·covariate]

The two raw coordinates still let the model exploit large-scale spatial trend; the
covariate and its interactions let it learn the covariate response. Tree ensembles
handle the mixed smooth/sharp structure without feature scaling and are fast and
deterministic on CPU (fixed ``random_state``). A small MLP is offered as an
alternative backbone for the curious, but the tree ensemble is the headline.

Honest caveat (kept front-and-centre)
-------------------------------------
This model produces a sharp point prediction but **no calibrated predictive-variance
surface** the way kriging does. If the deliverable is an uncertainty map (the usual
geostatistics requirement), that is a real advantage of the BEFORE method — see the
experiment README. Lower RMSE is not the whole story.

Contract (shared with the BEFORE side):

    run_after(dataset, ...) -> dict
        {"rmse":.., "mae":.., "y_pred":[...], "model":"...",
         "feature_importance": {...}}

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
    "FEATURE_NAMES",
    "build_features",
    "fit_ml",
    "run_after",
]

FEATURE_NAMES = ["x", "y", "covariate", "x*y", "covariate^2", "x*cov", "y*cov"]


def build_features(coords: np.ndarray, covariate: np.ndarray,
                   domain: float = 100.0) -> np.ndarray:
    """Assemble the design matrix [x, y, cov, x*y, cov^2, x*cov, y*cov].

    Coordinates and the covariate are lightly scaled (coords by the domain size, the
    covariate is left as-is since it is already standardised by the generator) so the
    engineered products are well-conditioned. Tree models do not need this, but it
    keeps the optional MLP backbone stable and the features human-readable.
    """
    coords = np.asarray(coords, float)
    cov = np.asarray(covariate, float).reshape(-1)
    x = coords[:, 0] / domain
    y = coords[:, 1] / domain
    feats = np.column_stack([
        x, y, cov,
        x * y,
        cov ** 2,
        x * cov,
        y * cov,
    ])
    return feats


def fit_ml(X: np.ndarray, y: np.ndarray, model: str = "rf",
           n_estimators: int = 300, max_depth: int | None = None,
           seed: int = 0):
    """Fit a deterministic sklearn regressor.

    ``model='rf'`` -> RandomForestRegressor (default headline).
    ``model='gbm'`` -> GradientBoostingRegressor.
    ``model='mlp'`` -> a small MLPRegressor (alternative neural backbone).
    All use a fixed ``random_state`` for reproducibility.
    """
    model = model.lower()
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(
            n_estimators=int(n_estimators), max_depth=max_depth,
            random_state=int(seed), n_jobs=-1,
        )
    elif model == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        reg = GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            max_depth=3 if max_depth is None else int(max_depth),
            learning_rate=0.05, subsample=0.9, random_state=int(seed),
        )
    elif model == "mlp":
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu",
            alpha=1e-3, max_iter=1500, random_state=int(seed),
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown model {model!r}")
    reg.fit(np.asarray(X, float), np.asarray(y, float))
    return reg


def run_after(dataset, model: str = "rf", n_estimators: int = 300,
              max_depth: int | None = None, seed: int = 0) -> dict:
    """Run the AFTER covariate-aware ML interpolator on a :class:`SpatialDataset`.

    Trains on station (coords + covariate) features and predicts on the dense grid,
    scoring against the grid's noise-free truth. Returns metrics + (for tree models)
    feature importances so the runner/README can show the covariate is being used.
    """
    p = dataset.params
    domain = getattr(p, "domain", 100.0)

    Xtr = build_features(dataset.train_coords, dataset.train_covariate, domain)
    ytr = dataset.train_value
    Xte = build_features(dataset.grid_coords, dataset.grid_covariate, domain)
    yte = dataset.grid_value

    reg = fit_ml(Xtr, ytr, model=model, n_estimators=n_estimators,
                 max_depth=max_depth, seed=seed)
    yp = reg.predict(Xte)

    importance = {}
    if hasattr(reg, "feature_importances_"):
        importance = {name: float(v) for name, v in
                      zip(FEATURE_NAMES, reg.feature_importances_)}

    return {
        "rmse": rmse(yte, yp),
        "mae": mae(yte, yp),
        "y_pred": np.asarray(yp, float).tolist(),
        "model": model,
        "n_estimators": int(n_estimators),
        "feature_importance": importance,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_spatial import synthetic_spatial_field

    ds = synthetic_spatial_field(n_points=160, seed=0, grid_res=30)
    for m in ("rf", "gbm"):
        res = run_after(ds, model=m, n_estimators=200)
        print(f"{m:4s} rmse={res['rmse']:.3f} mae={res['mae']:.3f}")
        if res["feature_importance"]:
            top = sorted(res["feature_importance"].items(),
                         key=lambda kv: -kv[1])[:3]
            print("     top features:", [(k, round(v, 3)) for k, v in top])
