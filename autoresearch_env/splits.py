"""Environmental-statistics split protocols.

Self-contained, lightweight versions of the splitters from the source framework
(``generalized_ml_autoresearch/core/evaluation/splits.py``), specialised for the
two leakage-resistant protocols env-stats forecasting cares about most:

- **WalkForwardSplit** — expanding-/rolling-window walk-forward, designed to be
  driven *by calendar year*: every fold tests on a future year never seen in
  training, with an optional ``gap`` to block target leakage at the seam. This is
  the honest evaluation for a warming, non-stationary climate series — the model
  must extrapolate forward in time, exactly as an operational forecast must.

- **SuperFoldSplit (by climate regime)** — a single train set that EXCLUDES every
  regime's held-out window at once, so a model is judged on its worst regime, not
  its average. Regimes here are synthetic climate states (e.g. ENSO-like
  warm/neutral/cool phases) supplied as an integer label per sample. This mirrors
  the FX "super-fold over regime windows" idea: don't let an easy regime mask a
  hard one.

Every splitter returns ``list[FoldAssignment]`` with the same shape the runner
consumes. ``validate_no_overlap`` programmatically enforces the data-integrity
invariant (no train/val/test row appears in two of the three within a fold).

Depends only on numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "FoldAssignment",
    "WalkForwardSplit",
    "SuperFoldSplit",
    "create_splitter",
    "validate_no_overlap",
    "SPLIT_REGISTRY",
]


@dataclass
class FoldAssignment:
    """One (train, val, test) index triple with an optional regime label."""

    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    regime_label: str = ""

    def n_train(self) -> int:
        return len(self.train_idx)

    def n_val(self) -> int:
        return len(self.val_idx)

    def n_test(self) -> int:
        return len(self.test_idx)


class _BaseSplitter:
    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:  # pragma: no cover
        raise NotImplementedError


class WalkForwardSplit(_BaseSplitter):
    """Walk-forward by contiguous blocks (intended: one block per year).

    The series is divided into ``n_blocks`` contiguous blocks. Fold ``k`` tests on
    block ``k + n_initial`` using all earlier blocks for training (the last
    ``val_fraction`` of the training span becomes validation). A ``gap`` of rows is
    purged between the end of train+val and the start of test to block leakage of a
    lagged target across the seam.

    Parameters
    ----------
    n_blocks : int
        Number of contiguous blocks the series is cut into (e.g. number of years).
    n_initial : int
        Blocks reserved for the first training window before the first test block.
    expanding : bool
        True -> training window grows each fold (expanding); False -> fixed-size
        rolling window of ``n_initial`` blocks.
    val_fraction : float
        Fraction of the (chronological) training span used as validation.
    gap : int
        Rows purged between the end of validation and the start of test.
    """

    def __init__(self, n_blocks: int = 5, n_initial: int = 2, expanding: bool = True,
                 val_fraction: float = 0.15, gap: int = 1):
        if n_initial < 1:
            raise ValueError("n_initial must be >= 1")
        if n_blocks <= n_initial:
            raise ValueError("n_blocks must exceed n_initial (need at least one test block)")
        self.n_blocks = n_blocks
        self.n_initial = n_initial
        self.expanding = expanding
        self.val_fraction = val_fraction
        self.gap = gap

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        # Block boundaries (as even as possible).
        edges = np.linspace(0, n_samples, self.n_blocks + 1).round().astype(int)
        out: list[FoldAssignment] = []
        for fold_id, test_block in enumerate(range(self.n_initial, self.n_blocks)):
            test_start, test_end = int(edges[test_block]), int(edges[test_block + 1])
            if self.expanding:
                train_block_start = 0
            else:
                train_block_start = test_block - self.n_initial
            trainval_start = int(edges[train_block_start])
            trainval_end = test_start - self.gap
            if trainval_end <= trainval_start:
                continue
            trainval = np.arange(trainval_start, trainval_end)
            n_val = max(1, int(round(len(trainval) * self.val_fraction)))
            train_idx = trainval[:-n_val]
            val_idx = trainval[-n_val:]
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            out.append(
                FoldAssignment(
                    fold_id, train_idx, val_idx, test_idx,
                    regime_label=f"walk-forward-block-{test_block}",
                )
            )
        return out


class SuperFoldSplit(_BaseSplitter):
    """Super-fold by climate regime: one fold whose train excludes every regime's hold-out.

    Each sample carries an integer regime label (``groups``). For every regime we
    hold out the last ``test_fraction`` of that regime's samples as test and the
    preceding ``val_fraction`` as validation; everything else is training. The
    result is ONE FoldAssignment whose test set spans ALL regimes, so the composite
    metric's per-regime penalty exposes a model that is strong on the common regime
    but weak on a rare one.

    Parameters
    ----------
    val_fraction, test_fraction : float
        Per-regime fractions (taken from the END of each regime's index order, so
        within a regime the test block is the most recent — no future leakage).
    label_horizon_buffer : int
        Rows immediately preceding each held-out block (per regime) that are also
        excluded from training, to block lagged-target leakage.
    """

    def __init__(self, val_fraction: float = 0.15, test_fraction: float = 0.15,
                 label_horizon_buffer: int = 0):
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.label_horizon_buffer = int(label_horizon_buffer)

    def split(self, n_samples: int, y=None, groups=None) -> list[FoldAssignment]:
        if groups is None:
            raise ValueError("SuperFoldSplit requires `groups` (a regime label per sample)")
        groups = np.asarray(groups)
        if len(groups) != n_samples:
            raise ValueError("groups length must equal n_samples")

        all_idx = np.arange(n_samples)
        val_mask = np.zeros(n_samples, dtype=bool)
        test_mask = np.zeros(n_samples, dtype=bool)
        excluded = np.zeros(n_samples, dtype=bool)

        for regime in np.unique(groups):
            r_idx = all_idx[groups == regime]
            r_idx = np.sort(r_idx)  # chronological within the regime
            n_r = len(r_idx)
            n_test = max(1, int(round(n_r * self.test_fraction)))
            n_val = max(1, int(round(n_r * self.val_fraction)))
            if n_test + n_val >= n_r:  # tiny regime: shrink so train is non-empty
                n_test = max(1, n_r // 3)
                n_val = max(1, n_r // 3)
            test_block = r_idx[n_r - n_test:]
            val_block = r_idx[n_r - n_test - n_val: n_r - n_test]
            test_mask[test_block] = True
            val_mask[val_block] = True
            excluded[test_block] = True
            excluded[val_block] = True
            if self.label_horizon_buffer > 0:
                first_held = (n_r - n_test - n_val)
                buf = r_idx[max(0, first_held - self.label_horizon_buffer): first_held]
                excluded[buf] = True

        train_idx = all_idx[~excluded]
        val_idx = all_idx[val_mask]
        test_idx = all_idx[test_mask]
        return [FoldAssignment(0, train_idx, val_idx, test_idx, regime_label="super-fold-by-regime")]


SPLIT_REGISTRY = {
    "walk_forward": WalkForwardSplit,
    "super_fold": SuperFoldSplit,
}


def create_splitter(name: str, **kwargs) -> _BaseSplitter:
    if name not in SPLIT_REGISTRY:
        raise ValueError(f"Unknown split protocol {name!r}; must be one of {list(SPLIT_REGISTRY)}")
    return SPLIT_REGISTRY[name](**kwargs)


def validate_no_overlap(fold_assignments: Iterable[FoldAssignment]) -> dict[str, int]:
    """Assert per-fold disjointness of train/val/test. Raises AssertionError on overlap.

    This is the programmatic enforcement of the data-integrity rule: no row may be
    in two of {train, val, test} within a fold.
    """
    report = {"folds_checked": 0, "total_overlaps": 0}
    for fa in fold_assignments:
        train = set(int(x) for x in fa.train_idx)
        val = set(int(x) for x in fa.val_idx)
        test = set(int(x) for x in fa.test_idx)
        tv, tt, vt = train & val, train & test, val & test
        if tv or tt or vt:
            report["total_overlaps"] += len(tv) + len(tt) + len(vt)
            raise AssertionError(
                f"Fold {fa.fold_id} overlap: train∩val={len(tv)}, "
                f"train∩test={len(tt)}, val∩test={len(vt)}"
            )
        report["folds_checked"] += 1
    return report
