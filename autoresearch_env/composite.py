"""Composite metric for the env-stats autoresearch loop, with Goodhart protection.

Mirrors the source framework's composite discipline
(``generalized_ml_autoresearch/core/evaluation/composite.py``):

    composite = min(val_primary, test_primary) - penalty_weight * n_below_threshold

For environmental forecasting the *primary* metric can be either:
  - an error metric where **lower is better** (latitude-weighted RMSE, RMSE, MAE),
    in which case we negate internally so a larger composite is always "better";
  - or a skill metric where **higher is better** (ACC, skill-vs-persistence).

The composite rewards the WORSE of validation/test (penalising overfit to either)
and subtracts a penalty for every fold whose primary metric is below an acceptance
threshold (penalising models that win on average but collapse on a regime).

**Frozen-fingerprint check (anti-Goodhart):** the metric configuration is hashed at
project start. If a later experiment is scored with a different configuration
(metric renamed, penalty changed, sign flipped, threshold moved), the fingerprint
changes and ``assert_fingerprint`` raises — catching a mid-project metric rewrite
that would silently make later results incomparable.

Stdlib only; runs anywhere.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

__all__ = ["CompositeCalculator", "MetricFingerprintError"]


class MetricFingerprintError(RuntimeError):
    """Raised when the composite-metric configuration changed mid-project."""


@dataclass
class CompositeCalculator:
    """Compute the composite score and guard its definition with a fingerprint.

    Parameters
    ----------
    primary_metric_name : str
        Name of the headline metric (e.g. "lat_weighted_rmse", "acc", "skill_vs_persistence").
    higher_is_better : bool
        True for skill metrics (ACC, skill score); False for error metrics (RMSE/MAE).
    penalty_weight : float
        Subtracted once per fold whose (oriented) primary metric is below threshold.
    below_threshold : float
        A fold counts as "bad" when its oriented primary metric < this value.
        Expressed in the SAME oriented sense the composite uses (higher == better),
        so for an error metric supply the negated cutoff (e.g. an RMSE cutoff of 12.0
        becomes below_threshold=-12.0).
    """

    primary_metric_name: str
    higher_is_better: bool = True
    penalty_weight: float = 0.1
    below_threshold: float = 0.0

    def _orient(self, value: float) -> float:
        """Return a 'higher is better' view of a raw metric value."""
        return value if self.higher_is_better else -value

    def compute(
        self,
        val_primary: float,
        test_primary: float,
        per_fold_test: Sequence[float],
    ) -> float:
        """composite = min(val, test) - penalty * n_below_threshold (oriented)."""
        v = self._orient(val_primary)
        t = self._orient(test_primary)
        fold_vals = [self._orient(x) for x in per_fold_test]
        n_below = sum(1 for x in fold_vals if x < self.below_threshold)
        return float(min(v, t) - self.penalty_weight * n_below)

    def fingerprint(self) -> str:
        """Stable hash of the metric definition — stored with every experiment."""
        raw = (
            f"{self.primary_metric_name}|{self.higher_is_better}|"
            f"{self.penalty_weight}|{self.below_threshold}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def assert_fingerprint(self, expected: str | None) -> None:
        """Raise if the live fingerprint differs from a previously frozen one.

        ``expected`` is the fingerprint recorded on the FIRST experiment of the
        project (read from best_config.json / experiment_log.jsonl). Passing None
        (no prior experiment) is a no-op.
        """
        if expected is None:
            return
        current = self.fingerprint()
        if current != expected:
            raise MetricFingerprintError(
                "Composite-metric fingerprint changed mid-project — refusing to score.\n"
                f"  frozen   = {expected}\n"
                f"  current  = {current}\n"
                f"  metric   = {self.primary_metric_name} "
                f"(higher_is_better={self.higher_is_better}, "
                f"penalty={self.penalty_weight}, threshold={self.below_threshold})\n"
                "If this change is intentional, record a RULE_CHANGE in the checkpoint "
                "and restart the experiment numbering — old composites are no longer comparable."
            )
