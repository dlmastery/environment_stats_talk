"""FAST unit tests for the foundation-encoder wrapper (Prithvi + timm fallback).

Run from the REPO ROOT::

    python -m pytest experiments/04_remote_sensing_landcover/tests -q

These tests guard the *contract* of ``after/foundation_encoder.py`` without
forcing a 471 MB Prithvi weight download or a real GPU. Specifically:

- the wrapper module imports cleanly,
- the band/normalisation constants match the Prithvi config.yaml (6 bands; mean
  and std are length-6 finite vectors at HLS reflectance scale, not [0,1]),
- ``PrithviAdapter.embed`` produces (N, 768) embeddings on a tiny synthetic
  batch using a randomly-initialised stand-in encoder (so the test is
  fully-offline and finishes in seconds),
- the generic timm fallback is skipped gracefully if ``timm`` is not installed
  or if the call to download ImageNet weights would touch the network.

We deliberately do NOT load the real Prithvi weights here — that download is
exercised by ``run_foundation.py`` and its committed JSON artefact.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_EXP_DIR = _TESTS_DIR.parent
_REPO_ROOT = _EXP_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make the experiment's ``after/`` dir importable as ``foundation_encoder``.
_AFTER = _EXP_DIR / "after"
if str(_AFTER) not in sys.path:
    sys.path.insert(0, str(_AFTER))


def _import_module():
    """Import the wrapper, skipping the whole test module if heavy deps missing."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"torch not importable: {e}")
    try:
        return importlib.import_module("foundation_encoder")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"foundation_encoder import failed: {e}")


# ---------------------------------------------------------------------------- #
# Import + constants                                                            #
# ---------------------------------------------------------------------------- #
def test_module_imports_and_exposes_api():
    fe = _import_module()
    for name in (
        "FoundationEncoderUnavailable", "PrithviAdapter",
        "load_prithvi_encoder", "encode_timm_vit",
        "extract_embeddings", "evaluate",
        "PRITHVI_HF_ID", "PRITHVI_BANDS", "PRITHVI_DATA_MEAN",
        "PRITHVI_DATA_STD", "PRITHVI_INPUT_SIZE", "PRITHVI_EMBED_DIM",
    ):
        assert hasattr(fe, name), f"foundation_encoder is missing {name!r}"


def test_prithvi_constants_match_published_config():
    fe = _import_module()
    # 6 HLS bands, 224-input, 768 ViT-B embedding.
    assert fe.PRITHVI_BANDS == ["B02", "B03", "B04", "B05", "B06", "B07"]
    assert fe.PRITHVI_INPUT_SIZE == 224
    assert fe.PRITHVI_EMBED_DIM == 768
    # data_mean/std are length-6 finite vectors in HLS reflectance units
    # (mean values are >>1 because HLS reflectance is scaled by 10000, not [0,1]).
    assert fe.PRITHVI_DATA_MEAN.shape == (6,)
    assert fe.PRITHVI_DATA_STD.shape == (6,)
    assert np.all(np.isfinite(fe.PRITHVI_DATA_MEAN))
    assert np.all(np.isfinite(fe.PRITHVI_DATA_STD))
    assert fe.PRITHVI_DATA_STD.min() > 0
    assert fe.PRITHVI_DATA_MEAN.min() > 100  # HLS scale, not [0,1]


# ---------------------------------------------------------------------------- #
# Adapter embedding shape (no real weights — offline, fast)                     #
# ---------------------------------------------------------------------------- #
def test_prithvi_adapter_embed_shape_offline():
    """Verify the 5→6 band remap, upsample, and CLS-token extraction wire correctly.

    We instantiate the real ``PrithviViT`` class with random weights (no download)
    so the test is fully offline and finishes in a few seconds. The point is the
    *plumbing*, not the model's quality.
    """
    fe = _import_module()
    import torch
    try:
        from huggingface_hub import hf_hub_download
        prithvi_py = hf_hub_download(fe.PRITHVI_HF_ID, "prithvi_mae.py")
    except Exception as e:
        pytest.skip(f"prithvi_mae.py not available offline: {e}")
    prithvi_dir = Path(prithvi_py).parent
    if str(prithvi_dir) not in sys.path:
        sys.path.insert(0, str(prithvi_dir))
    try:
        prithvi_mod = importlib.import_module("prithvi_mae")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not import prithvi_mae: {e}")

    # Random-init encoder (no real weights downloaded; pure plumbing test).
    model = prithvi_mod.PrithviViT(
        img_size=fe.PRITHVI_INPUT_SIZE,
        patch_size=(1, 16, 16),
        num_frames=1, in_chans=6,
        embed_dim=fe.PRITHVI_EMBED_DIM, depth=2, num_heads=12,
        encoder_only=True,
    ).eval()
    adapter = fe.PrithviAdapter(model=model, device=torch.device("cpu"),
                                 batch_size=2)

    # Tiny synthetic batch in our 5-band [0,1] reflectance format.
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 0.6, size=(4, 5, 8, 8)).astype(np.float32)
    embs = adapter.embed(X)
    assert embs.shape == (4, fe.PRITHVI_EMBED_DIM), embs.shape
    assert np.all(np.isfinite(embs))


# ---------------------------------------------------------------------------- #
# timm fallback path: skip cleanly if not available / offline                   #
# ---------------------------------------------------------------------------- #
def test_timm_fallback_skips_when_unavailable():
    fe = _import_module()
    try:
        import timm  # noqa: F401
    except ImportError:
        pytest.skip("timm not installed (the fallback is intentionally optional)")
    # If timm IS installed we still skip downloading ImageNet weights here — that
    # is exercised by run_foundation.py --encoder timm, not by CI.
    pytest.skip("timm available but ImageNet weight download is skipped in unit tests")
