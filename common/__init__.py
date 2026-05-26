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
)

__all__ = [
    "metrics", "plotting",
    "daily_temperature", "daily_precipitation", "gridded_temperature_field", "time_split",
    "citizen_comments", "gold_interactions", "SPECIES", "RELATIONS",
    "multispectral_patches", "compute_indices", "change_pair", "BANDS", "CLASSES",
]

__version__ = "0.1.0"
