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
    "HARD_CLASSES", "multispectral_patches_hard",
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


# --------------------------------------------------------------------------- #
# HARD MODE — texture-separable classes with (near-)identical mean spectra.    #
# --------------------------------------------------------------------------- #
# Motivation: in the default `multispectral_patches`, every class has a distinct
# mean spectral signature, so the BEFORE pipeline (RandomForest on NDVI/NDWI/
# brightness, which *average* each patch to a few scalars) already separates them
# near-perfectly. That makes the AFTER CNN's edge a tie, not a win.
#
# `multispectral_patches_hard` fixes this by building classes whose *mean* spectra
# are (nearly) identical within texture pairs, so any feature that averages over
# the patch — NDVI, NDWI, mean SWIR, brightness, i.e. exactly what `compute_indices`
# returns — is blind to them. They differ ONLY in SPATIAL TEXTURE / pattern, which
# a convolutional model can read but a mean-index Random Forest structurally cannot:
#
#   class                base spectrum   distinguishing spatial pattern
#   -------------------  --------------  --------------------------------------------
#   forest_natural       "forest"        isotropic, low-frequency mottle (natural canopy)
#   orchard_rows         "forest"        oriented, periodic ROWS (planted orchard/vineyard)
#   water_smooth         "water"         isotropic wind-ripple texture (open water)
#   flooded_field        "water"         oriented ROWS from submerged crop rows (flooded field)
#   bare                 "bare"          isotropic mid-frequency texture (spectral anchor class)
#
# `forest_natural` vs `orchard_rows` share the SAME mean greenness (same NDVI/NDWI);
# `water_smooth` vs `flooded_field` share the SAME mean water spectrum. Within each
# pair the two textures are deliberately matched in amplitude AND spatial-correlation
# magnitude — they differ only in ISOTROPY vs ORIENTATION (rows). That matters: if a
# pair differed in spatial-correlation length (e.g. smooth vs white speckle), the
# patch-mean of the nonlinear NDVI ratio would pick up a tell-tale *variance*
# difference and a tree ensemble could exploit it. Matching the correlation magnitude
# keeps mean indices (and any per-patch summary of them) at chance, so resolving the
# pair genuinely requires reading the oriented spatial pattern — a CNN's job.
# The two pairs differ from each other and from `bare` spectrally, so RF can still
# tell the *pairs* apart; it just cannot resolve *within* a pair, where the CNN wins.
HARD_CLASSES = [
    "forest_natural", "orchard_rows", "water_smooth", "flooded_field", "bare",
]
# Each hard class reuses a base mean spectrum from `_SIGNATURES`. Texture pairs
# deliberately share a base, so their per-patch MEAN spectra coincide.
_HARD_BASE = {
    "forest_natural": "forest",
    "orchard_rows":   "forest",
    "water_smooth":   "water",
    "flooded_field":  "water",
    "bare":           "bare",
}


def _zero_mean(field: np.ndarray) -> np.ndarray:
    """Remove the per-patch mean so a texture pattern adds NO spectral-mean signal.

    This is the crux of hard mode: the texture is added as a zero-mean perturbation,
    so two patches of a texture pair have identical band means (hence identical
    NDVI/NDWI/brightness) and differ only in spatial structure.
    """
    return field - field.mean()


def _box_blur(a: np.ndarray, k: int) -> np.ndarray:
    """Separable box blur (a tiny dependency-free low-pass filter), shape-preserving.

    Used to shape white Gaussian noise into different spatial-frequency textures
    while keeping the marginal (per-pixel) distribution Gaussian and symmetric, so
    that two textures normalised to the same variance have the SAME per-pixel
    histogram and therefore identical expected band means / NDVI / NDWI.
    """
    if k <= 1:
        return a.astype(np.float32)
    kern = np.ones(k, dtype=np.float32) / k
    out = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 0, a)
    out = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 1, out)
    return out.astype(np.float32)


def _texture_field(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a zero-mean, unit-std (size, size) float32 spatial-texture field.

    Every field is built by *spatially filtering Gaussian white noise* (so the
    marginal stays Gaussian and symmetric) then standardised to zero-mean / unit-std.
    Two fields of the same amplitude therefore share an identical per-pixel
    distribution and differ ONLY in spatial frequency / orientation — which mean
    indices (NDVI/NDWI/brightness) are blind to but convolutions are not.
    """
    base = rng.standard_normal((size, size)).astype(np.float32)
    if kind == "isotropic_low":
        # smooth, low-frequency mottle (natural canopy): strong low-pass blur.
        field = _box_blur(base, k=max(3, size // 3))
    elif kind == "rows":
        # periodic ROWS (planted orchard/vineyard): blur strongly ALONG one random
        # axis and not across it -> oriented stripes (anisotropic high freq).
        ang = rng.integers(0, 2)  # 0 = horizontal rows, 1 = vertical rows
        a = base if ang == 0 else base.T
        kern = np.ones(max(3, size // 2), dtype=np.float32)
        kern /= kern.sum()
        a = np.apply_along_axis(lambda m: np.convolve(m, kern, mode="same"), 1, a)
        field = a if ang == 0 else a.T
    elif kind == "flat":
        # near-flat (calm water): very strong low-pass -> almost no structure.
        field = _box_blur(base, k=size)
    elif kind == "speckle":
        # high-frequency salt-and-pepper SPECKLE (flooded field): raw white noise.
        field = base
    elif kind == "isotropic_mid":
        # mid-frequency isotropic texture (bare soil clods): mild low-pass.
        field = _box_blur(base, k=3)
    else:  # pragma: no cover - guarded by HARD_CLASSES
        raise ValueError(f"unknown texture kind: {kind}")
    field = field.astype(np.float32)
    std = float(field.std())
    if std > 1e-6:
        field = field / std
    return _zero_mean(field)


# Texture "kind" + amplitude per hard class. Within each spectral pair the two
# members are matched in amplitude AND in spatial-correlation magnitude (both use a
# correlated, equal-strength field) and differ ONLY in ISOTROPY vs ORIENTATION, so
# the distinction is purely spatial pattern — invisible to any mean/variance summary
# of the indices, learnable by a CNN. (`flat`/`speckle` remain available below but
# are intentionally unused: a smooth-vs-white-speckle pair leaks an NDVI-variance
# tell that a tree ensemble can exploit; see the module-level note.)
_HARD_TEXTURE = {
    "forest_natural": ("isotropic_low", 0.060),
    "orchard_rows":   ("rows",          0.060),
    "water_smooth":   ("isotropic_mid", 0.040),
    "flooded_field":  ("rows",          0.040),
    "bare":           ("isotropic_mid", 0.060),
}


def multispectral_patches_hard(n: int = 600, size: int = 16, seed: int = 0):
    """Return (X, y) for the *texture-separable* HARD land-cover task.

    X: float32 array (n, 5, size, size) of reflectance in ~[0,1].
    y: int array (n,) of class indices into ``HARD_CLASSES``.

    Unlike :func:`multispectral_patches`, several classes share a (near-)identical
    MEAN spectrum and differ ONLY in spatial texture (orchard rows vs natural forest;
    smooth water vs speckled flooded field). Mean-index features (NDVI/NDWI/
    brightness, i.e. :func:`compute_indices`) therefore cannot separate the texture
    pairs — a convolutional model that reads spatial pattern can. This is the regime
    where a CNN-on-bands honestly beats RandomForest-on-indices.

    The texture is added as a per-band, zero-mean spatial field (see
    :func:`_texture_field` / :func:`_zero_mean`), so within a texture pair the
    per-patch band means coincide by construction.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n, len(BANDS), size, size), dtype=np.float32)
    y = rng.integers(0, len(HARD_CLASSES), size=n)
    grad = np.linspace(-0.015, 0.015, size, dtype=np.float32)
    gx, gygrid = np.meshgrid(grad, grad)
    base_illum = _zero_mean((gx + gygrid).astype(np.float32))
    for i in range(n):
        cls = HARD_CLASSES[y[i]]
        sig = np.array(_SIGNATURES[_HARD_BASE[cls]], dtype=np.float32)
        kind, amp = _HARD_TEXTURE[cls]
        # A faint zero-mean illumination tilt (shared form, random strength) so the
        # patches are not perfectly flat; it carries no mean-spectral signal.
        illum = base_illum * rng.uniform(0.3, 1.0)
        for b in range(len(BANDS)):
            tex = _texture_field(kind, size, rng) * amp
            X[i, b] = sig[b] + illum + tex
    X = np.clip(X, 0.0, 1.0)
    return X, y.astype(int)


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
