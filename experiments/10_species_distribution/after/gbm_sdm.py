"""AFTER - SDM via sklearn GradientBoostingClassifier (automatic interactions).

The AFTER method consumes **the same columns** as the BEFORE GLM
(``temperature``, ``precipitation``, ``elevation``) but does NOT need the
analyst to hand-craft squared terms, interaction terms or the right basis. A
gradient-boosted tree ensemble learns those interactions automatically -- it
discovers the Gaussian niche (a non-monotone curve in T and P) and the small
elevation interaction baked into the synthetic generator without being told.

Why this is a fair AFTER baseline (not a free lunch)
----------------------------------------------------
- Same data, same train/test split, same prevalence as the GLM.
- Sklearn-only, CPU-only, no GPU; ``GradientBoostingClassifier`` is in every
  scientific Python install.
- Modest configuration (n_estimators=200, max_depth=3, learning_rate=0.05).
  The numbers are not tuned per dataset; they are a sensible default that
  recovers nonlinear niches without overfitting at the sample sizes used here.
- The model returns calibrated-ish probabilities via the sklearn implementation
  of the boosted-logit objective (``deviance``).

Outputs from :func:`run_after` mirror the BEFORE module's output so the runner
can compare them apples-to-apples.

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

__all__ = [
    "FEATURES",
    "build_features",
    "fit_gbm",
    "run_after",
]

FEATURES = ("temperature", "precipitation", "elevation")


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Stack the raw covariate columns (no standardization needed for trees)."""
    return np.column_stack([df[f].to_numpy(dtype=float) for f in FEATURES])


def fit_gbm(X_train: np.ndarray, y_train: np.ndarray,
            n_estimators: int = 200,
            max_depth: int = 3,
            learning_rate: float = 0.05,
            random_state: int = 0,
            calibrate: bool = True,
            calibration_cv: int = 3):
    """Fit a GradientBoostingClassifier, optionally wrapped in isotonic calibration.

    Calibration is a standard SDM hygiene step: raw boosted trees rank well
    (good AUC) but their probability outputs can be over-confident, which hurts
    Brier. ``CalibratedClassifierCV(method="isotonic", cv=K)`` fits the GBM on
    K cross-validation folds and learns a monotone mapping on the held-out
    predictions, producing calibrated probabilities while preserving the ranking
    that the trees recover. No tuning per dataset; just a defensible default.
    """
    base = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        subsample=1.0,
    )
    if calibrate:
        # cv=K: GBM is internally refit K times on (K-1)/K of the data, and the
        # held-out fold is used to fit the isotonic mapping. Final predict_proba
        # averages the K calibrated estimators.
        model = CalibratedClassifierCV(base, method="isotonic", cv=calibration_cv)
    else:
        model = base
    model.fit(X_train, y_train)
    return model


def run_after(records_train: pd.DataFrame, records_test: pd.DataFrame,
              grid_df: pd.DataFrame,
              n_estimators: int = 200,
              max_depth: int = 3,
              learning_rate: float = 0.05,
              random_state: int = 0,
              calibrate: bool = True) -> dict:
    """Fit the AFTER GBM on TRAIN, predict on TEST and on the grid.

    Inputs have the same schema as :func:`before.glm_logistic.run_before`.
    """
    X_tr = build_features(records_train)
    y_tr = records_train["y_label"].to_numpy(dtype=int)
    X_te = build_features(records_test)
    X_grid = build_features(grid_df)

    model = fit_gbm(X_tr, y_tr,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    calibrate=calibrate)

    p_tr = model.predict_proba(X_tr)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    p_grid = model.predict_proba(X_grid)[:, 1]

    # Feature importances: CalibratedClassifierCV averages over CV folds, each
    # with its own GBM estimator. For reporting we average the underlying GBM
    # importances across folds (if calibrated) or read them directly (if not).
    importances = _extract_feature_importances(model)

    return {
        "method": "gbm_sdm",
        "feature_names": list(FEATURES),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "calibrated": bool(calibrate),
        "p_pres_train": p_tr,
        "p_pres_test": p_te,
        "p_grid": p_grid,
        "y_train": y_tr,
        "y_test": records_test["y_label"].to_numpy(dtype=int),
        "feature_importances": dict(zip(
            FEATURES, [float(v) for v in importances])),
        "model": model,
    }


def _extract_feature_importances(model) -> np.ndarray:
    """Average the underlying GBM feature_importances_ across calibration folds.

    Works for both a bare ``GradientBoostingClassifier`` and a
    ``CalibratedClassifierCV`` wrapping one.
    """
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if isinstance(model, CalibratedClassifierCV):
        vals = []
        for ccv in model.calibrated_classifiers_:
            inner = getattr(ccv, "estimator", None) or getattr(ccv, "base_estimator", None)
            if inner is not None and hasattr(inner, "feature_importances_"):
                vals.append(np.asarray(inner.feature_importances_, dtype=float))
        if vals:
            return np.mean(np.vstack(vals), axis=0)
    return np.full(len(FEATURES), np.nan)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_sdm import synthetic_sdm_dataset

    ds = synthetic_sdm_dataset(n_pres=400, n_bg=2000, seed=0)
    rec = ds.records_df
    n = len(rec)
    cut = int(0.7 * n)
    tr, te = rec.iloc[:cut], rec.iloc[cut:]
    out = run_after(tr, te, ds.grid_df)
    print("GBM feature importances:")
    for k, v in out["feature_importances"].items():
        print(f"  {k:>15s}: {v:.4f}")
    print(f"mean p(test): {out['p_pres_test'].mean():.3f}  "
          f"prevalence: {te['y_label'].mean():.3f}")
