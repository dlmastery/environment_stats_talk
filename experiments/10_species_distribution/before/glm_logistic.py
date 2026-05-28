"""BEFORE - classical SDM as a logistic regression / MaxEnt-style GLM.

This is what most ecologists / environmental statisticians actually fit first:
a logistic regression of presence vs background on a small, *hand-crafted*
feature set:

    linear     : temperature, precipitation, elevation
    squared    : temperature^2, precipitation^2

Squared terms are included because the textbook intuition is that a Gaussian
niche should be representable as a quadratic in the centred climate variables
(``log P(suit) ~ -(T - T*)^2 / (2 sT^2) - (P - P*)^2 / (2 sP^2)`` is exactly
a quadratic when expanded), so the GLM with quadratics is the *fair* classical
baseline: it captures a univariate Gaussian niche but cannot, without further
manual crafting, recover

  * the *unknown* niche centres (the optimizer fits the quadratic, fine), or
  * the *interaction* between elevation and the precipitation-niche width that
    is baked into the synthetic generator's true surface.

So a plain GLM with linear + squared terms is the BEFORE method whose error
floor is **structural** (wrong basis), not noise.

Outputs from :func:`run_before`:

    p_pres_test    : predicted P(presence) on the held-out evaluation rows
    p_grid         : predicted P(presence) on the evaluation grid (for the map)
    model_summary  : coefficient names + values for reporting
    feature_names  : feature-column order

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

__all__ = [
    "BASE_FEATURES",
    "build_glm_features",
    "fit_glm",
    "run_before",
]

BASE_FEATURES = ("temperature", "precipitation", "elevation")
SQUARED_FEATURES = ("temperature", "precipitation")  # add squared terms for these


def build_glm_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the GLM design matrix (linear + a couple of squared terms).

    The squared terms are added in their raw scale; the caller is expected to
    standardize the columns (see ``fit_glm``). Returns (X, feature_names).
    """
    parts = []
    names: list[str] = []
    for f in BASE_FEATURES:
        parts.append(df[f].to_numpy(dtype=float)[:, None])
        names.append(f)
    for f in SQUARED_FEATURES:
        parts.append(df[f].to_numpy(dtype=float)[:, None] ** 2)
        names.append(f + "^2")
    X = np.hstack(parts)
    return X, names


def fit_glm(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0,
            random_state: int = 0):
    """Fit a standardized logistic regression (the GLM).

    Returns (model, scaler) so the same scaling can be applied to evaluation
    inputs. ``C`` is the inverse L2 regularization strength (sklearn convention);
    we keep mild regularization (``C=1.0``) so the GLM is not artificially
    weakened.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    # L2 is the sklearn default; specifying penalty="l2" is now deprecated.
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )
    model.fit(Xs, y_train)
    return model, scaler


def run_before(records_train: pd.DataFrame, records_test: pd.DataFrame,
               grid_df: pd.DataFrame, C: float = 1.0,
               random_state: int = 0) -> dict:
    """Fit the BEFORE GLM on TRAIN, predict on TEST and on the grid.

    The training and test tables must each have the columns ``temperature``,
    ``precipitation``, ``elevation`` and the binary target ``y_label``. The grid
    table has the three covariates plus the noise-free ``suitability`` column
    used for the suitability-correlation score by the runner.
    """
    X_tr, feat_names = build_glm_features(records_train)
    y_tr = records_train["y_label"].to_numpy(dtype=int)
    X_te, _ = build_glm_features(records_test)
    X_grid, _ = build_glm_features(grid_df)

    model, scaler = fit_glm(X_tr, y_tr, C=C, random_state=random_state)

    p_tr = model.predict_proba(scaler.transform(X_tr))[:, 1]
    p_te = model.predict_proba(scaler.transform(X_te))[:, 1]
    p_grid = model.predict_proba(scaler.transform(X_grid))[:, 1]

    coefs = dict(zip(feat_names, [float(v) for v in model.coef_[0]]))
    coefs["intercept"] = float(model.intercept_[0])

    return {
        "method": "glm_logistic",
        "feature_names": feat_names,
        "C": float(C),
        "p_pres_train": p_tr,
        "p_pres_test": p_te,
        "p_grid": p_grid,
        "y_train": y_tr,
        "y_test": records_test["y_label"].to_numpy(dtype=int),
        "coefficients": coefs,
        "model": model,
        "scaler": scaler,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_sdm import synthetic_sdm_dataset

    ds = synthetic_sdm_dataset(n_pres=400, n_bg=2000, seed=0)
    rec = ds.records_df
    n = len(rec)
    cut = int(0.7 * n)
    tr, te = rec.iloc[:cut], rec.iloc[cut:]
    out = run_before(tr, te, ds.grid_df)
    print("GLM coefficients:")
    for k, v in out["coefficients"].items():
        print(f"  {k:>18s}: {v:+.4f}")
    print(f"mean p(test): {out['p_pres_test'].mean():.3f}  "
          f"prevalence: {te['y_label'].mean():.3f}")
