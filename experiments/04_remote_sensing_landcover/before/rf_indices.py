"""BEFORE — the classic GIS-analyst land-cover workflow.

This is what a remote-sensing analyst does *before* deep learning: compute a
handful of physically-motivated spectral indices per patch (NDVI for vegetation,
NDWI for open water, mean SWIR for soil/moisture, overall brightness), then train
a tabular classifier (Random Forest) on those few features.

It is fast, transparent, runs on a laptop CPU, and it is a genuinely strong
baseline on well-separated classes -- which is exactly why it is the honest
reference the AFTER CNN (on the raw 5-band patches, capturing *texture*) must beat.

Contract (so the orchestrator can treat BEFORE and AFTER uniformly):

    fit_rf(X_train, y_train, seed) -> fitted sklearn RandomForestClassifier
    predict(model, X) -> y_pred (int labels into common.CLASSES)
    run_before(X_train, y_train, X_test, y_test, seed) -> metrics dict

The model is trained on ``compute_indices(X)`` features, never on raw pixels.

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make `import common` work whether invoked via pytest (rootdir on path) or as a
# script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import compute_indices, metrics, CLASSES  # noqa: E402

__all__ = ["INDEX_NAMES", "fit_rf", "predict", "run_before"]

INDEX_NAMES = ["NDVI", "NDWI", "mean_SWIR", "brightness"]


def fit_rf(X_train: np.ndarray, y_train: np.ndarray, seed: int = 0,
           n_estimators: int = 200, max_depth: int | None = None):
    """Fit a Random Forest on hand-computed spectral indices.

    ``X_train`` is the raw (N,5,H,W) patch array; we reduce it to the (N,4) index
    table with ``compute_indices`` before fitting. Returns the fitted estimator.
    """
    from sklearn.ensemble import RandomForestClassifier

    feats = compute_indices(X_train)
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(feats, y_train)
    return clf


def predict(model, X: np.ndarray) -> np.ndarray:
    """Predict land-cover labels for raw patches via their spectral indices."""
    feats = compute_indices(X)
    return model.predict(feats).astype(int)


def run_before(X_train, y_train, X_test, y_test, seed: int = 0,
               n_estimators: int = 200) -> dict:
    """Train RF-on-indices and evaluate on the held-out test patches.

    Returns a metrics dict::

        {"accuracy":.., "macro_f1":.., "macro_precision":.., "macro_recall":..,
         "feature_importance": {index_name: importance, ...},
         "y_pred":[...], "n_train":.., "n_test":..}
    """
    model = fit_rf(X_train, y_train, seed=seed, n_estimators=n_estimators)
    y_pred = predict(model, X_test)
    rep = metrics.classification_report_simple(y_test, y_pred)
    importances = {
        name: float(imp)
        for name, imp in zip(INDEX_NAMES, model.feature_importances_)
    }
    return {
        "model": "RandomForest-on-indices",
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        "macro_precision": rep["macro_precision"],
        "macro_recall": rep["macro_recall"],
        "feature_importance": importances,
        "y_pred": np.asarray(y_pred, int).tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import multispectral_patches

    Xtr, ytr = multispectral_patches(n=600, size=16, seed=0)
    Xte, yte = multispectral_patches(n=200, size=16, seed=1)
    res = run_before(Xtr, ytr, Xte, yte, seed=0)
    print("== BEFORE: RandomForest on spectral indices ==")
    print(f"  classes: {CLASSES}")
    print(f"  accuracy : {res['accuracy']:.3f}")
    print(f"  macro-F1 : {res['macro_f1']:.3f}")
    print("  feature importance:")
    for name, imp in res["feature_importance"].items():
        print(f"    {name:10s} {imp:.3f}")
