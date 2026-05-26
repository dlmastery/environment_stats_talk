"""Synthetic multispectral remote-sensing patch generator.

Generates small image patches with 5 bands ordered [blue, green, red, nir, swir]
and a land-cover label. Each class has a characteristic spectral signature plus
spatial texture, so that:
- BEFORE: a Random Forest on hand-computed indices (NDVI/NDWI) does reasonably,
- AFTER: a small CNN on raw bands (4090) captures texture and does better,
and so NDVI/NDWI-based change detection is meaningful.

Swap-in note: replace `multispectral_patches` with real Sentinel-2 chips
(bands B2,B3,B4,B8,B11) loaded via rasterio/sentinelhub; keep the (N, C, H, W)
float array in [0,1] reflectance and the integer label vector.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "BANDS", "CLASSES", "multispectral_patches", "compute_indices", "change_pair",
]

BANDS = ["blue", "green", "red", "nir", "swir"]
CLASSES = ["water", "forest", "cropland", "urban", "bare"]

# Mean reflectance signature per class, per band (rough, illustrative).
_SIGNATURES = {
    "water":    [0.06, 0.05, 0.04, 0.02, 0.01],
    "forest":   [0.03, 0.06, 0.04, 0.45, 0.18],
    "cropland": [0.06, 0.10, 0.09, 0.38, 0.26],
    "urban":    [0.18, 0.19, 0.20, 0.24, 0.30],
    "bare":     [0.20, 0.24, 0.30, 0.34, 0.40],
}
_TEXTURE = {  # spatial heterogeneity per class (urban/cropland are patchier)
    "water": 0.010, "forest": 0.030, "cropland": 0.055, "urban": 0.080, "bare": 0.040,
}


def multispectral_patches(n: int = 600, size: int = 16, seed: int = 0):
    """Return (X, y).

    X: float32 array (n, 5, size, size) of reflectance in ~[0,1].
    y: int array (n,) of class indices into ``CLASSES``.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(BANDS), size, size), dtype=np.float32)
    y = rng.integers(0, len(CLASSES), size=n)
    for i in range(n):
        cls = CLASSES[y[i]]
        sig = np.array(_SIGNATURES[cls], dtype=np.float32)
        tex = _TEXTURE[cls]
        # base signature + smooth gradient (illumination) + per-pixel texture
        grad = np.linspace(-0.02, 0.02, size, dtype=np.float32)
        gx, gygrid = np.meshgrid(grad, grad)
        for b in range(len(BANDS)):
            patch = sig[b] + (gx + gygrid) * rng.uniform(0.3, 1.0)
            patch = patch + rng.normal(0, tex, size=(size, size)).astype(np.float32)
            X[i, b] = patch
    X = np.clip(X, 0.0, 1.0)
    return X, y.astype(int)


def compute_indices(X: np.ndarray) -> np.ndarray:
    """Per-patch spectral indices used by the BEFORE pipeline.

    Returns array (n, 4): [NDVI, NDWI, mean_swir, brightness] averaged per patch.
    - NDVI = (nir - red) / (nir + red)         vegetation
    - NDWI = (green - nir) / (green + nir)      open water
    """
    eps = 1e-6
    blue, green, red, nir, swir = (X[:, b] for b in range(5))
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    brightness = X.mean(axis=1)
    feats = np.stack([
        ndvi.mean(axis=(1, 2)),
        ndwi.mean(axis=(1, 2)),
        swir.mean(axis=(1, 2)),
        brightness.mean(axis=(1, 2)),
    ], axis=1)
    return feats.astype(np.float32)


def change_pair(n: int = 300, size: int = 16, seed: int = 0, change_frac: float = 0.3):
    """Generate (X_t0, X_t1, changed) for change detection.

    ``change_frac`` of patches transition forest -> bare/cropland (deforestation);
    the rest are unchanged forest/cropland. ``changed`` is a 0/1 label per patch.
    """
    rng = np.random.default_rng(seed)
    changed = (rng.random(n) < change_frac).astype(int)
    X0 = np.zeros((n, len(BANDS), size, size), dtype=np.float32)
    X1 = np.zeros_like(X0)
    for i in range(n):
        start = "forest"
        end = ("bare" if rng.random() < 0.5 else "cropland") if changed[i] else "forest"
        for arr, cls in ((X0, start), (X1, end)):
            sig = np.array(_SIGNATURES[cls], dtype=np.float32)
            tex = _TEXTURE[cls]
            for b in range(len(BANDS)):
                arr[i, b] = np.clip(
                    sig[b] + rng.normal(0, tex, size=(size, size)).astype(np.float32),
                    0.0, 1.0)
    return X0, X1, changed
