"""FAST unit tests for the Experiment 04 HARD path (CPU, tiny n, 1-2 epochs).

Run from the REPO ROOT::

    python -m pytest experiments/04_remote_sensing_landcover/tests -q

These guard the hard-mode *contracts* (not the headline win, which needs the GPU
run): the orchestrator runs end-to-end on tiny CPU data, the BEFORE/AFTER metrics
are finite and in range, all artifacts are written under ``results/hard/``, and the
hard generator's defining property — that mean-index features are NOT separable on
a texture pair while raw patches are — holds on a small sample.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

_TESTS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _TESTS_DIR.parent
_REPO_ROOT = _EXP_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import generator DIRECTLY from the module (not via common/__init__).
from common.synthetic_remote_sensing import (  # noqa: E402
    multispectral_patches_hard, compute_indices, HARD_CLASSES, BANDS,
)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _EXP_DIR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cnn_hard = _load("th_cnn_hard", "after/cnn_hard.py")

N_CLASSES = len(HARD_CLASSES)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# ------------------------------- generator --------------------------------- #
def test_hard_generator_shape_range():
    X, y = multispectral_patches_hard(n=24, size=8, seed=0)
    assert X.shape == (24, len(BANDS), 8, 8)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert set(np.unique(y)).issubset(set(range(N_CLASSES)))


def test_hard_indices_blind_to_a_texture_pair():
    """A linear classifier on mean indices stays near chance on a texture pair."""
    from sklearn.linear_model import LogisticRegression
    a = HARD_CLASSES.index("forest_natural")
    b = HARD_CLASSES.index("orchard_rows")
    Xtr, ytr = multispectral_patches_hard(n=400, size=12, seed=0)
    Xte, yte = multispectral_patches_hard(n=200, size=12, seed=1)
    mtr, mte = np.isin(ytr, [a, b]), np.isin(yte, [a, b])
    clf = LogisticRegression(max_iter=1000).fit(compute_indices(Xtr[mtr]), ytr[mtr])
    acc = clf.score(compute_indices(Xte[mte]), yte[mte])
    assert acc < 0.70  # mean indices cannot resolve the texture pair


# ------------------------------ orchestrator ------------------------------- #
def test_cnn_hard_quick_runs_and_writes(tmp_path):
    out = cnn_hard.main([
        "--quick", "--cpu", "--epochs", "1", "--seed", "0",
        "--results-dir", str(tmp_path),
    ])
    cls = out["classification"]
    for blk in (cls["before_rf_indices"], cls["after_cnn_bands"]):
        assert _finite(blk["accuracy"]) and _finite(blk["macro_f1"])
        assert 0.0 <= blk["accuracy"] <= 1.0
    assert _finite(cls["delta"]["accuracy"])
    assert out["classes"] == HARD_CLASSES
    assert out["classification"]["after_cnn_bands"]["device"] == "cpu"
    for fn in ("metrics.json", "confusion_matrix_hard.png",
               "before_after_hard_bars.png", "summary.md"):
        assert (tmp_path / fn).exists(), fn
