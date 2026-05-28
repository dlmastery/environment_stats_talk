"""FAST unit tests for Experiment 10 (species distribution modelling).

CPU-only, no torch, no network. The full suite finishes in well under 30 s
(a few sklearn fits on a small synthetic SDM dataset).

What is asserted (the honest BEFORE-vs-AFTER claims):
- Generator wiring: presence + background classes are both present in the
  combined records table; shapes match.
- Metrics are finite (no NaN AUC / Brier).
- On a small held-out test set, the AFTER GBM beats the BEFORE GLM on AUC,
  and its Brier score is no worse than the GLM's (Brier <= GLM Brier).
- True-suitability correlation: GBM is at least as well-correlated with the
  noise-free truth as the GLM (the GLM with only linear+squared terms cannot
  represent the elevation-driven interaction in the synthetic niche).

Run from the REPO ROOT:
    python -m pytest experiments/10_species_distribution/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, brier_score_loss

# Repo root (for `import common`) and the experiment dir (for before/after pkgs).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXP_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.synthetic_sdm import synthetic_sdm_dataset  # noqa: E402
from before.glm_logistic import run_before, build_glm_features  # noqa: E402
from after.gbm_sdm import run_after, build_features  # noqa: E402


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_split():
    """Tiny SDM dataset with a deterministic train/test split.

    Sizes chosen so the whole module runs in a couple of seconds while still
    being large enough for AUC / Brier to be stable.
    """
    ds = synthetic_sdm_dataset(n_pres=250, n_bg=1500, seed=0)
    rng = np.random.default_rng(101)
    idx = np.arange(len(ds.records_df))
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    train = ds.records_df.iloc[idx[:cut]].reset_index(drop=True)
    test = ds.records_df.iloc[idx[cut:]].reset_index(drop=True)
    return ds, train, test


# --------------------------------------------------------------------------- #
def test_split_has_both_classes(small_split):
    _, train, test = small_split
    assert train["y_label"].nunique() == 2
    assert test["y_label"].nunique() == 2


def test_glm_design_matrix_shape(small_split):
    _, train, _ = small_split
    X, names = build_glm_features(train)
    # 3 linear + 2 squared = 5 columns
    assert X.shape == (len(train), 5)
    assert names == ["temperature", "precipitation", "elevation",
                     "temperature^2", "precipitation^2"]
    assert np.isfinite(X).all()


def test_gbm_design_matrix_shape(small_split):
    _, train, _ = small_split
    X = build_features(train)
    assert X.shape == (len(train), 3)
    assert np.isfinite(X).all()


def test_metrics_finite_for_both_methods(small_split):
    ds, train, test = small_split
    before = run_before(train, test, ds.grid_df)
    after = run_after(train, test, ds.grid_df,
                      n_estimators=150, max_depth=3, learning_rate=0.05)
    y_te = test["y_label"].to_numpy(dtype=int)
    for out in (before, after):
        auc = roc_auc_score(y_te, out["p_pres_test"])
        brier = brier_score_loss(y_te, out["p_pres_test"])
        suit_corr = np.corrcoef(
            ds.grid_df["suitability"].to_numpy(), out["p_grid"])[0, 1]
        assert np.isfinite(auc)
        assert np.isfinite(brier)
        assert np.isfinite(suit_corr)
        # probability outputs are in [0, 1]
        assert out["p_pres_test"].min() >= 0.0
        assert out["p_pres_test"].max() <= 1.0
        assert out["p_grid"].min() >= 0.0
        assert out["p_grid"].max() <= 1.0


def test_ml_beats_glm_on_auc(small_split):
    """The AFTER GBM should rank presences above background better than the GLM."""
    ds, train, test = small_split
    before = run_before(train, test, ds.grid_df)
    after = run_after(train, test, ds.grid_df,
                      n_estimators=150, max_depth=3, learning_rate=0.05)
    y_te = test["y_label"].to_numpy(dtype=int)
    auc_glm = roc_auc_score(y_te, before["p_pres_test"])
    auc_gbm = roc_auc_score(y_te, after["p_pres_test"])
    assert auc_gbm > auc_glm, (
        f"GBM AUC {auc_gbm:.4f} did not exceed GLM AUC {auc_glm:.4f}")


def test_ml_brier_no_worse_than_glm(small_split):
    """Calibrated GBM should produce Brier no worse than the GLM."""
    ds, train, test = small_split
    before = run_before(train, test, ds.grid_df)
    after = run_after(train, test, ds.grid_df,
                      n_estimators=150, max_depth=3, learning_rate=0.05)
    y_te = test["y_label"].to_numpy(dtype=int)
    brier_glm = brier_score_loss(y_te, before["p_pres_test"])
    brier_gbm = brier_score_loss(y_te, after["p_pres_test"])
    # Allow a tiny slack for finite-sample noise; the headline claim is "<= GLM".
    assert brier_gbm <= brier_glm + 1e-3, (
        f"GBM Brier {brier_gbm:.4f} not <= GLM Brier {brier_glm:.4f}")


def test_ml_recovers_true_suitability_at_least_as_well(small_split):
    """The GLM with only linear+squared terms cannot represent the elevation
    interaction baked into the synthetic niche; the GBM should correlate with
    the noise-free truth at least as well as the GLM."""
    ds, train, test = small_split
    before = run_before(train, test, ds.grid_df)
    after = run_after(train, test, ds.grid_df,
                      n_estimators=150, max_depth=3, learning_rate=0.05)
    suit = ds.grid_df["suitability"].to_numpy()
    r_glm = np.corrcoef(suit, before["p_grid"])[0, 1]
    r_gbm = np.corrcoef(suit, after["p_grid"])[0, 1]
    assert r_gbm >= r_glm - 1e-3, (
        f"GBM suitability_corr {r_gbm:.4f} fell well below GLM {r_glm:.4f}")
