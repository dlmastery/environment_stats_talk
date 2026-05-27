"""Shared utilities for the environment_stats_talk experiments.

Deterministic synthetic data generators + metrics + headless plotting, so every
experiment's committed results are reproducible offline (no API keys, no GPU).
"""
from __future__ import annotations

from . import metrics, plotting
from .synthetic_climate import (
    daily_temperature, daily_precipitation, gridded_temperature_field, time_split,
)
from .synthetic_biodiversity import citizen_comments, gold_interactions, SPECIES, RELATIONS
from .synthetic_remote_sensing import (
    multispectral_patches, compute_indices, change_pair, BANDS, CLASSES,
    multispectral_patches_hard, HARD_CLASSES,
)
from .synthetic_hydrology import synthetic_catchment
from .synthetic_spatial import synthetic_spatial_field
from .synthetic_airquality import synthetic_pm25

__all__ = [
    "metrics", "plotting",
    # climate
    "daily_temperature", "daily_precipitation", "gridded_temperature_field", "time_split",
    # biodiversity
    "citizen_comments", "gold_interactions", "SPECIES", "RELATIONS",
    # remote sensing (incl. texture-separable hard mode)
    "multispectral_patches", "compute_indices", "change_pair", "BANDS", "CLASSES",
    "multispectral_patches_hard", "HARD_CLASSES",
    # hydrology / spatial / air quality
    "synthetic_catchment", "synthetic_spatial_field", "synthetic_pm25",
]

__version__ = "0.1.0"
