"""autoresearch_env — a self-contained, lightweight env-stats autoresearch loop.

Adapted from the user's own ``dlmastery/autoresearch`` (``generalized_ml_autoresearch``):
the 7-step Diagnose -> Cite -> Hypothesize -> Predict -> Execute -> Analyze ->
Checkpoint loop, with hard gates (citation rigor + reasoning completeness), an
env composite metric with a frozen Goodhart fingerprint, and env-flavored splits
(walk-forward by year, super-fold by climate regime).

It deliberately does NOT import the source repo; it reuses this repo's ``common``
package for synthetic data + metrics so everything runs offline on CPU.
"""
from __future__ import annotations

from .reasoning import (
    ReasoningEntry,
    ReasoningAnnotationsFile,
    validate_citation_rigor,
    validate_reasoning_blob,
    validate_pre_run_entry,
)
from .composite import CompositeCalculator, MetricFingerprintError
from .splits import (
    FoldAssignment,
    WalkForwardSplit,
    SuperFoldSplit,
    create_splitter,
    validate_no_overlap,
)
from .runner import run_experiment, load_dataset, ExperimentRecord

__all__ = [
    "ReasoningEntry",
    "ReasoningAnnotationsFile",
    "validate_citation_rigor",
    "validate_reasoning_blob",
    "validate_pre_run_entry",
    "CompositeCalculator",
    "MetricFingerprintError",
    "FoldAssignment",
    "WalkForwardSplit",
    "SuperFoldSplit",
    "create_splitter",
    "validate_no_overlap",
    "run_experiment",
    "load_dataset",
    "ExperimentRecord",
]

__version__ = "0.1.0"
