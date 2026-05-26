"""FAST unit tests for Experiment 04 (CPU, tiny n, 1 epoch; target < 30s).

Run from the REPO ROOT::

    python -m pytest experiments/04_remote_sensing_landcover/tests -q

These guard the *contracts*, not the headline accuracy:
- BEFORE: RandomForest-on-indices fits and beats random chance (1/n_classes) on
  the well-separated synthetic classes.
- AFTER: the CNN forward pass produces the right shape, and one training step (1
  epoch) runs and yields finite, valid metrics.
- Change detector returns a finite, in-range F1.
- The orchestrator's metrics are all finite (via a --quick, 1-epoch run).
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

from common import multispectral_patches, change_pair, CLASSES  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _EXP_DIR / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf_indices = _load("t_rf_indices", "before/rf_indices.py")
cnn_classifier = _load("t_cnn_classifier", "after/cnn_classifier.py")
change_detect = _load("t_change_detect", "after/change_detect.py")
runner = _load("t_run_before_after", "run_before_after.py")

N_CLASSES = len(CLASSES)
CHANCE = 1.0 / N_CLASSES


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


# ------------------------------- BEFORE (RF) ------------------------------- #
def test_rf_fits_and_beats_chance():
    Xtr, ytr = multispectral_patches(n=160, size=8, seed=0)
    Xte, yte = multispectral_patches(n=80, size=8, seed=1)
    res = rf_indices.run_before(Xtr, ytr, Xte, yte, seed=0, n_estimators=40)
    assert _finite(res["accuracy"]) and _finite(res["macro_f1"])
    assert 0.0 <= res["macro_f1"] <= 1.0
    # well-separated spectral signatures => clearly above random chance
    assert res["accuracy"] > CHANCE + 0.1
    assert set(res["feature_importance"]) == set(rf_indices.INDEX_NAMES)


# -------------------------------- AFTER (CNN) ------------------------------ #
def test_cnn_forward_shape():
    model = cnn_classifier.SmallCNN(width=8)
    x = torch.randn(4, 5, 8, 8)
    out = model(x)
    assert out.shape == (4, N_CLASSES)


def test_cnn_one_train_step_runs():
    Xtr, ytr = multispectral_patches(n=120, size=8, seed=0)
    device = torch.device("cpu")  # tests pinned to CPU
    model, history = cnn_classifier.train_cnn(
        Xtr, ytr, epochs=1, batch_size=32, width=8, patience=1,
        seed=0, device=device,
    )
    assert len(history["train_loss"]) == 1
    assert _finite(history["train_loss"][0])
    y_pred = cnn_classifier.predict(model, Xtr[:16], device=device)
    assert y_pred.shape == (16,)
    assert set(np.unique(y_pred)).issubset(set(range(N_CLASSES)))


def test_cnn_run_after_finite():
    Xtr, ytr = multispectral_patches(n=120, size=8, seed=0)
    Xte, yte = multispectral_patches(n=60, size=8, seed=1)
    res = cnn_classifier.run_after(
        Xtr, ytr, Xte, yte, epochs=1, batch_size=32, width=8,
        patience=1, seed=0, device=torch.device("cpu"),
    )
    assert _finite(res["accuracy"]) and _finite(res["macro_f1"])
    assert 0.0 <= res["accuracy"] <= 1.0
    assert res["n_params"] > 0
    assert res["device"] == "cpu"


def test_embedding_knn_baseline_skips_cleanly():
    # No encoder supplied -> documented skip, never a fabricated number.
    Xtr, ytr = multispectral_patches(n=20, size=8, seed=0)
    out = cnn_classifier.embedding_knn_baseline(Xtr, ytr, Xtr, ytr, encoder=None)
    assert out["skipped"] is True
    assert "how_to_enable" in out


def test_embedding_knn_baseline_with_dummy_encoder():
    # A trivial encoder (the 4 indices) proves the kNN path is wired correctly.
    from common import compute_indices
    Xtr, ytr = multispectral_patches(n=120, size=8, seed=0)
    Xte, yte = multispectral_patches(n=60, size=8, seed=1)
    out = cnn_classifier.embedding_knn_baseline(
        Xtr, ytr, Xte, yte, encoder=lambda x: compute_indices(np.asarray(x)), k=5)
    assert out["skipped"] is False
    assert _finite(out["accuracy"]) and _finite(out["macro_f1"])


# ----------------------------- change detection ---------------------------- #
def test_change_detector_finite_f1():
    X0, X1, changed = change_pair(n=120, size=8, seed=0, change_frac=0.3)
    res = change_detect.run_change(X0, X1, changed)
    assert _finite(res["accuracy"]) and _finite(res["macro_f1"])
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert _finite(res["threshold"])
    assert len(res["ndvi_drop"]) == len(changed)


# ------------------------------ orchestrator ------------------------------- #
def test_runner_quick_metrics_finite(tmp_path):
    out = runner.main([
        "--quick", "--cpu", "--epochs", "1", "--seed", "0",
        "--results-dir", str(tmp_path),
    ])
    cls = out["classification"]
    for blk in (cls["before_rf_indices"], cls["after_cnn_bands"]):
        assert _finite(blk["accuracy"]) and _finite(blk["macro_f1"])
    assert _finite(out["change_detection"]["macro_f1"])
    # artifacts written
    for fn in ("metrics.json", "confusion_matrix.png", "change_map.png",
               "before_after_bars.png", "summary.md"):
        assert (tmp_path / fn).exists(), fn
