"""Change detection — NDVI-difference thresholding on a before/after image pair.

The simplest, most interpretable change detector in the remote-sensing toolbox:
compute NDVI for time t0 and t1, take the per-patch mean drop in NDVI, and flag a
patch as "changed" when the drop exceeds a threshold. For deforestation
(forest -> bare/cropland) NDVI falls sharply, so a single threshold already
separates changed from unchanged well.

We keep it transparent on purpose -- this is the AFTER side's *baseline* change
detector (a Siamese CNN would be the heavy-weight upgrade, noted in the README),
yet it already gives a finite, reportable accuracy/F1 against the ``changed``
labels, and a per-patch change *map* we can render.

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics  # noqa: E402

__all__ = ["patch_ndvi", "ndvi_diff", "best_threshold", "detect_change", "run_change"]

# Band order is [blue, green, red, nir, swir]; NDVI uses red (idx 2) and nir (3).
_RED, _NIR = 2, 3


def patch_ndvi(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-patch mean NDVI for an (N,5,H,W) reflectance cube -> (N,) array."""
    red = X[:, _RED]
    nir = X[:, _NIR]
    ndvi = (nir - red) / (nir + red + eps)
    return ndvi.mean(axis=(1, 2))


def ndvi_diff(X0: np.ndarray, X1: np.ndarray) -> np.ndarray:
    """NDVI *drop* from t0 to t1 (positive = vegetation loss) -> (N,) array."""
    return patch_ndvi(X0) - patch_ndvi(X1)


def best_threshold(diff: np.ndarray, changed: np.ndarray) -> float:
    """Pick the NDVI-drop threshold maximising macro-F1 on a labelled set.

    Scans candidate thresholds (midpoints between sorted unique diffs). In a real
    pipeline this threshold is calibrated on a small labelled validation set; here
    we expose it so the demo can show the calibration step explicitly.
    """
    diff = np.asarray(diff, float)
    changed = np.asarray(changed, int)
    cands = np.unique(diff)
    if len(cands) > 1:
        mids = (cands[:-1] + cands[1:]) / 2.0
        cands = np.concatenate([[cands[0] - 1e-3], mids, [cands[-1] + 1e-3]])
    best_t, best_f1 = float(cands[0]), -1.0
    for t in cands:
        pred = (diff >= t).astype(int)
        f1 = metrics.classification_report_simple(changed, pred)["macro_f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def detect_change(X0: np.ndarray, X1: np.ndarray, threshold: float) -> np.ndarray:
    """Flag patches whose NDVI drop meets/exceeds ``threshold`` (0/1 per patch)."""
    return (ndvi_diff(X0, X1) >= threshold).astype(int)


def run_change(X0, X1, changed, threshold: float | None = None) -> dict:
    """Run NDVI-difference change detection and score it against ``changed``.

    If ``threshold`` is None it is calibrated on this set (max macro-F1). Returns a
    metrics dict with accuracy, macro-F1, the threshold used, and the per-patch
    NDVI drop + predictions (used to render the change map).
    """
    diff = ndvi_diff(X0, X1)
    thr = best_threshold(diff, changed) if threshold is None else float(threshold)
    pred = (diff >= thr).astype(int)
    rep = metrics.classification_report_simple(changed, pred)
    return {
        "task": "NDVI-difference change detection",
        "threshold": float(thr),
        "accuracy": rep["accuracy"],
        "macro_f1": rep["macro_f1"],
        "ndvi_drop": np.asarray(diff, float).tolist(),
        "y_pred": np.asarray(pred, int).tolist(),
        "y_true": np.asarray(changed, int).tolist(),
        "n": int(len(changed)),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import change_pair

    X0, X1, changed = change_pair(n=300, size=16, seed=0, change_frac=0.3)
    res = run_change(X0, X1, changed)
    print("== Change detection: NDVI-difference threshold ==")
    print(f"  threshold : {res['threshold']:.3f}")
    print(f"  accuracy  : {res['accuracy']:.3f}")
    print(f"  macro-F1  : {res['macro_f1']:.3f}  (n={res['n']})")
