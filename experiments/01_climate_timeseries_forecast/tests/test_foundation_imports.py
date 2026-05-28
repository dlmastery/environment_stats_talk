"""FAST import + shape sanity tests for the Exp01 foundation-model wrappers.

These tests DO NOT download model weights (that's an integration test). They
only verify that:

  1. The wrapper module under ``after/<model>_zero_shot.py`` imports cleanly
     when its backing package is installed; otherwise we ``pytest.skip``.
  2. The wrapper exposes the expected public surface (``evaluate``, ``predict``).
  3. With a tiny synthetic series, the ``_predict_batch`` shim returns the
     correct output shape — gated behind ``RUN_FOUNDATION_DOWNLOAD=1`` so CI
     can stay offline-only by default.

Run from the REPO ROOT:
    python -m pytest experiments/01_climate_timeseries_forecast/tests -q
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

AFTER_DIR = _EXP_DIR / "after"
ALLOW_DOWNLOAD = os.environ.get("RUN_FOUNDATION_DOWNLOAD", "0") == "1"


def _load_after_module(filename: str):
    p = AFTER_DIR / filename
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# (wrapper_filename, required_pip_package, optional fallback transformers attr)
WRAPPERS = [
    ("chronos_zero_shot.py", "chronos", None),
    ("timesfm_zero_shot.py", None, "TimesFmModelForPrediction"),
    ("moment_zero_shot.py", "momentfm", None),
    ("moirai_zero_shot.py", "uni2ts", None),
]


@pytest.mark.parametrize("wrapper, pkg, hf_attr", WRAPPERS)
def test_wrapper_imports_cleanly(wrapper, pkg, hf_attr):
    """Every wrapper must import without error when its backing pkg is present."""
    if pkg is not None:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            pytest.skip(f"{pkg} not installed: {e}")
    if hf_attr is not None:
        try:
            from transformers import TimesFmModelForPrediction  # noqa: F401
        except ImportError as e:
            pytest.skip(f"transformers.{hf_attr} unavailable: {e}")
    mod = _load_after_module(wrapper)
    assert hasattr(mod, "evaluate"), f"{wrapper} missing evaluate()"
    assert callable(mod.evaluate), f"{wrapper}.evaluate not callable"
    # Every wrapper must expose a batched-predict helper. The Chronos wrapper
    # predates this convention and exposes ``_predict_batch`` instead; both
    # are acceptable for the shape-only smoke test below.
    assert hasattr(mod, "predict") or hasattr(mod, "_predict_batch"), (
        f"{wrapper} missing predict() / _predict_batch() helper"
    )


@pytest.mark.parametrize("wrapper, pkg, hf_attr", WRAPPERS)
def test_predict_shape_smoke(wrapper, pkg, hf_attr):
    """Tiny synthetic shape check.

    Gated behind ``RUN_FOUNDATION_DOWNLOAD=1`` because building a real model
    triggers a HuggingFace download on first run. Set the env var locally to
    exercise; CI / cold tests default-skip.
    """
    if not ALLOW_DOWNLOAD:
        pytest.skip("set RUN_FOUNDATION_DOWNLOAD=1 to exercise weights download")
    if pkg is not None:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            pytest.skip(f"{pkg} not installed: {e}")
    if hf_attr is not None:
        try:
            from transformers import TimesFmModelForPrediction  # noqa: F401
        except ImportError as e:
            pytest.skip(f"transformers.{hf_attr} unavailable: {e}")

    mod = _load_after_module(wrapper)

    # synthetic mini-batch: 2 series of length 64
    rng = np.random.default_rng(0)
    contexts = [rng.standard_normal(64).astype(np.float32) for _ in range(2)]
    h = 4

    if wrapper == "chronos_zero_shot.py":
        pipe, _name, mode = mod._load_pipeline(prefer="bolt")
        import torch
        ctxs_t = [torch.from_numpy(c).float() for c in contexts]
        out = mod._predict_batch(pipe, mode, ctxs_t, prediction_length=h)
    elif wrapper == "timesfm_zero_shot.py":
        model, _name, _device = mod._load_model()
        out = mod.predict(model, contexts, h)
    elif wrapper == "moment_zero_shot.py":
        model, _name, _device = mod._load_model()
        # MOMENT needs contexts of length ``MOMENT_INPUT_LEN - h`` minimum;
        # extend the 64-long synthetic ones by repeating to satisfy that.
        L = mod.MOMENT_INPUT_LEN - h
        contexts = [np.tile(c, int(np.ceil(L / len(c))))[:L] for c in contexts]
        out = mod.predict(model, contexts, h)
    elif wrapper == "moirai_zero_shot.py":
        model, _device, _ctx, pred = mod.build_for_test(
            context_length=64, prediction_length=h
        )
        out = mod.predict(model, contexts, pred, num_samples=4)
    else:
        pytest.fail(f"no test branch for {wrapper}")

    assert out.shape[0] == 2, f"{wrapper} returned batch={out.shape[0]} (want 2)"
    assert out.shape[1] >= h, (
        f"{wrapper} returned horizon={out.shape[1]} (want >= {h})"
    )
    assert np.isfinite(out).all(), f"{wrapper} returned non-finite values"
