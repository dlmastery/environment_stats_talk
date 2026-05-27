"""Tests for the HARD (texture-separable) remote-sensing generator.

Run from the repo root:  python -m pytest common/tests -q

These guard the *defining property* of `multispectral_patches_hard`:

- shapes / dtype / value range are correct, labels index into ``HARD_CLASSES``;
- the texture-pair classes share (near-)identical MEAN spectra, so MEAN-INDEX
  features (``compute_indices`` -> NDVI/NDWI/SWIR/brightness) are *not*
  class-separable on them: a linear classifier (LDA / logistic regression) scores
  near chance on the two texture-only pairs;
- the information IS present in the raw patches: simple translation-invariant
  TEXTURE statistics (spatial std + oriented gradient energy, the kind of signal a
  CNN learns) separate the very same pairs near-perfectly. This is the sanity check
  that the task is solvable from spatial pattern, just not from mean indices.

We import the generator DIRECTLY from ``common.synthetic_remote_sensing`` (not via
``common/__init__``), per the experiment's import convention.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from common.synthetic_remote_sensing import (
    multispectral_patches_hard,
    compute_indices,
    HARD_CLASSES,
    BANDS,
)

# Texture pairs that share a mean spectrum (indices: forest pair / water pair).
_FOREST_PAIR = (HARD_CLASSES.index("forest_natural"),
                HARD_CLASSES.index("orchard_rows"))
_WATER_PAIR = (HARD_CLASSES.index("water_smooth"),
               HARD_CLASSES.index("flooded_field"))


def _texture_stats(X: np.ndarray) -> np.ndarray:
    """Translation-invariant per-band texture descriptors (a CNN-style signal).

    Returns (N, 4*5): spatial std, mean horizontal-gradient energy, mean
    vertical-gradient energy, and |horiz - vert| anisotropy, per band. These are
    blind to absolute level (they describe *pattern*, not mean spectrum).
    """
    X = np.asarray(X, dtype=np.float32)
    sd = X.std(axis=(2, 3))
    dh = np.abs(np.diff(X, axis=3)).mean(axis=(2, 3))
    dv = np.abs(np.diff(X, axis=2)).mean(axis=(2, 3))
    aniso = np.abs(dh - dv)
    return np.concatenate([sd, dh, dv, aniso], axis=1)


def _pair_acc(feat_fn, clf_factory, a: int, b: int) -> float:
    """Held-out binary accuracy separating classes ``a`` vs ``b`` (chance = 0.5)."""
    Xtr, ytr = multispectral_patches_hard(n=1000, size=16, seed=0)
    Xte, yte = multispectral_patches_hard(n=500, size=16, seed=1)
    mtr, mte = np.isin(ytr, [a, b]), np.isin(yte, [a, b])
    clf = clf_factory().fit(feat_fn(Xtr[mtr]), ytr[mtr])
    return float(clf.score(feat_fn(Xte[mte]), yte[mte]))


# ------------------------------- shapes / range --------------------------- #
def test_hard_shape_range_labels():
    X, y = multispectral_patches_hard(n=40, size=8, seed=0)
    assert X.shape == (40, len(BANDS), 8, 8)
    assert X.dtype == np.float32
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)).issubset(set(range(len(HARD_CLASSES))))


def test_hard_deterministic():
    a = multispectral_patches_hard(n=30, size=8, seed=3)[0]
    b = multispectral_patches_hard(n=30, size=8, seed=3)[0]
    assert np.array_equal(a, b)


# ---------------- mean spectra coincide within texture pairs --------------- #
def test_texture_pairs_share_mean_spectrum():
    X, y = multispectral_patches_hard(n=1500, size=16, seed=0)
    means = X.mean(axis=(2, 3))  # (N, 5) per-patch band means
    for a, b in (_FOREST_PAIR, _WATER_PAIR):
        ma = means[y == a].mean(axis=0)
        mb = means[y == b].mean(axis=0)
        # band means agree to well under a percent of reflectance
        assert np.max(np.abs(ma - mb)) < 0.01, (HARD_CLASSES[a], HARD_CLASSES[b], ma, mb)


# ------- index features are NOT separable on the texture-only pairs -------- #
def test_indices_near_chance_on_texture_pairs():
    for a, b in (_FOREST_PAIR, _WATER_PAIR):
        acc = _pair_acc(compute_indices, LinearDiscriminantAnalysis, a, b)
        # LDA on mean indices cannot beat chance by a meaningful margin
        assert acc < 0.62, (HARD_CLASSES[a], HARD_CLASSES[b], acc)


def test_indices_logreg_also_near_chance_on_texture_pairs():
    for a, b in (_FOREST_PAIR, _WATER_PAIR):
        acc = _pair_acc(compute_indices, lambda: LogisticRegression(max_iter=2000),
                        a, b)
        assert acc < 0.62, (HARD_CLASSES[a], HARD_CLASSES[b], acc)


# --------- but raw-patch TEXTURE statistics ARE separable (sanity) --------- #
def test_texture_stats_separate_pairs():
    for a, b in (_FOREST_PAIR, _WATER_PAIR):
        acc = _pair_acc(_texture_stats, lambda: LogisticRegression(max_iter=2000),
                        a, b)
        assert acc > 0.90, (HARD_CLASSES[a], HARD_CLASSES[b], acc)


def test_full_task_indices_capped_textures_unlocked():
    """End-to-end contrast: mean-index ceiling is well below the texture ceiling."""
    Xtr, ytr = multispectral_patches_hard(n=1000, size=16, seed=0)
    Xte, yte = multispectral_patches_hard(n=500, size=16, seed=1)
    idx_acc = LinearDiscriminantAnalysis().fit(
        compute_indices(Xtr), ytr).score(compute_indices(Xte), yte)
    tex_acc = LogisticRegression(max_iter=2000).fit(
        _texture_stats(Xtr), ytr).score(_texture_stats(Xte), yte)
    assert idx_acc < 0.75            # indices cannot fully solve the 5-class task
    assert tex_acc > 0.90            # texture-aware features can
    assert tex_acc - idx_acc > 0.20  # a clear, honest gap
