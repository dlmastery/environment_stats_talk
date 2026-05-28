"""Import-only smoke tests for the climate / weather foundation-model stack.

These do NOT download anything and DO NOT run a model forward pass. They
only verify that each library installed in this env imports without
errors, and skip cleanly otherwise. The full real-run scripts live next
to this file under each model's directory.
"""
from __future__ import annotations

import importlib

import pytest


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # pragma: no cover -- import-only test
        pytest.skip(f"{name} not importable: {e!r}")


def test_graphcast_imports():
    g = _import("graphcast.graphcast")
    assert hasattr(g, "GraphCast"), "GraphCast class missing"
    assert hasattr(g, "TARGET_SURFACE_VARS"), "TARGET_SURFACE_VARS missing"
    assert len(g.TARGET_SURFACE_VARS) == 5
    assert "2m_temperature" in g.TARGET_SURFACE_VARS


def test_graphcast_pressure_levels():
    g = _import("graphcast.graphcast")
    levels = g.PRESSURE_LEVELS_WEATHERBENCH_13
    assert len(levels) == 13
    assert 50 in levels and 1000 in levels


def test_neuralgcm_imports():
    ng = _import("neuralgcm")
    assert hasattr(ng, "PressureLevelModel"), "PressureLevelModel missing"
    # The classmethod we use for loading.
    assert hasattr(ng.PressureLevelModel, "from_checkpoint")


def test_dinosaur_imports():
    d = _import("dinosaur")
    assert d is not None


def test_onnxruntime_imports():
    ort = _import("onnxruntime")
    assert "CPUExecutionProvider" in ort.get_available_providers()


def test_climax_imports():
    # Only run if the ClimaX clone is on sys.path (the run.py adds it
    # at runtime; the smoke test just checks the package import path).
    pytest.importorskip("climax", reason="ClimaX source not on sys.path")
    a = _import("climax.arch")
    assert hasattr(a, "ClimaX"), "ClimaX class missing"


def test_fourcastnet_imports():
    # FourCastNet ships as a repo, not a pip package -- only check timm
    # which the AFNO backbone needs.
    timm = _import("timm")
    assert hasattr(timm, "__version__")
    einops = _import("einops")
    assert hasattr(einops, "__version__")


def test_truststore_imports():
    ts = _import("truststore")
    assert hasattr(ts, "inject_into_ssl")


def test_common_synthetic_climate():
    # Should always succeed: this is our own utility.
    sc = _import("common.synthetic_climate")
    g = sc.gridded_temperature_field(n_lat=8, n_lon=16, seed=0)
    assert g.field.shape == (8, 16)
    # Equator should be warmer than poles (sanity).
    import numpy as np
    assert g.field[g.lats.argmin()].mean() < g.field[len(g.lats) // 2].mean()
