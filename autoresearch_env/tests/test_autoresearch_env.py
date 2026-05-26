"""Fast unit tests for the env-stats autoresearch loop.

Run from the repo root:
    python -m pytest autoresearch_env/tests -q

These are intentionally FAST (synthetic data, numpy MLP, a handful of years) so the
build stays green in seconds. They verify the three load-bearing behaviours:

1. The reasoning gate REJECTS a missing/shallow entry and ACCEPTS a complete one.
2. The composite fingerprint DETECTS a mid-project metric rewrite.
3. A 2-3 experiment mini-loop runs on synthetic data, and the champion record only
   updates when the composite IMPROVES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `import autoresearch_env` / `import common` resolve from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autoresearch_env.reasoning import (  # noqa: E402
    ReasoningEntry,
    ReasoningAnnotationsFile,
    validate_pre_run_entry,
    validate_reasoning_blob,
)
from autoresearch_env.composite import CompositeCalculator, MetricFingerprintError  # noqa: E402
from autoresearch_env.splits import (  # noqa: E402
    WalkForwardSplit,
    SuperFoldSplit,
    validate_no_overlap,
)
from autoresearch_env.runner import run_experiment, load_dataset  # noqa: E402


# ----------------------------- reasoning fixtures ---------------------------

def _complete_entry(exp_num: int = 1) -> ReasoningEntry:
    """A full, gate-passing pre-run reasoning entry (real env-stats citation)."""
    return ReasoningEntry(
        experiment_num=exp_num,
        diagnosis=(
            "This is the baseline one-step-ahead temperature-forecasting experiment on the "
            "synthetic daily 2m-temperature series, which carries an annual cycle, a multidecadal "
            "warming trend, and AR(1) weather persistence. No prior champion exists, so the "
            "diagnosis is scope-setting: establish a defensible walk-forward RMSE floor that any "
            "later neural or foundation-model experiment must out-skill, evaluated per calendar "
            "block so the model is judged on future years it never trained on."
        ),
        citations=(
            "Hyndman & Athanasopoulos 2021 'Forecasting: Principles and Practice' (3rd ed., OTexts) "
            "— motivates persistence as the skill reference because a forecast that cannot beat "
            "'tomorrow equals today' has no operational value, so we report skill-vs-persistence "
            "and use a walk-forward-by-year protocol to respect the warming trend's non-stationarity "
            "exactly as an operational environmental forecast would be scored in practice."
        ),
        hypothesis=(
            "We hypothesize that a tiny 1-hidden-layer MLP on 7 autoregressive lags plus a "
            "sin/cos day-of-year encoding will reach a walk-forward test RMSE near the persistence "
            "floor because the mechanism is that short-lag autocorrelation plus the seasonal "
            "encoding captures most of the one-step-ahead signal per Gardner & Dorling 1998's "
            "atmospheric-MLP review; gains over persistence should be modest at horizon one."
        ),
        prediction=(
            "Walk-forward test RMSE should land in the range 1.5 to 3.0 deg C across folds, with "
            "skill-vs-persistence in the range -0.05 to +0.15 (i.e. roughly matching to slightly "
            "beating the naive forecast at one-step-ahead)."
        ),
    )


# ----------------------------- gate behaviour -------------------------------

def test_gate_rejects_missing_entry(tmp_path):
    """No reasoning entry at all -> runner refuses to launch."""
    cfg = _base_config(tmp_path, backbone="persistence")
    with pytest.raises(ValueError, match="no pre-run reasoning annotation"):
        run_experiment(cfg)


def test_gate_rejects_shallow_entry(tmp_path):
    """A too-short / no-citation entry fails the pre-run gate with multiple violations."""
    shallow = ReasoningEntry(
        experiment_num=1,
        diagnosis="too short",
        citations="(ClimateLLM2025)",   # bare parenthetical tag, no venue/title/note
        hypothesis="try the mlp",        # no mechanism keyword, below floor
        prediction="better",            # no numeric range
    )
    violations = validate_pre_run_entry(shallow)
    assert len(violations) >= 4
    assert any("diagnosis" in v for v in violations)
    assert any("citations" in v for v in violations)
    assert any("numeric range" in v for v in violations)


def test_gate_accepts_complete_entry(tmp_path):
    """A complete entry passes the pre-run gate cleanly."""
    assert validate_pre_run_entry(_complete_entry()) == []


def test_full_blob_needs_verdict_and_learning():
    """The post-run blob validator additionally requires a KEEP/DISCARD verdict + learning."""
    e = _complete_entry()
    # pre-run passes, but the full blob is missing verdict/learning
    assert validate_pre_run_entry(e) == []
    full_violations = validate_reasoning_blob(e)
    assert any("verdict" in v.lower() for v in full_violations)
    assert any("learning" in v.lower() for v in full_violations)

    e.verdict = (
        "KEEP. Composite improved to a new champion; all walk-forward folds beat the "
        "persistence reference and the worst fold still cleared the acceptance threshold "
        "comfortably, so the baseline is confirmed as the floor for later experiments."
    )
    e.learning = (
        "Baseline confirmed — axis open for an LSTM with longer memory. Next try: a recurrent "
        "backbone on a 30-day window to test whether longer-memory persistence beats the 7-lag "
        "MLP, since the warming trend and AR(1) weather structure may reward more history than a "
        "one-week window can encode at the one-step-ahead horizon we evaluate here."
    )
    assert validate_reasoning_blob(e) == []


# ----------------------------- composite + fingerprint ----------------------

def test_composite_min_and_penalty():
    """composite = min(val,test) - penalty*n_below, oriented for error metrics."""
    calc = CompositeCalculator("lat_weighted_rmse", higher_is_better=False,
                               penalty_weight=0.5, below_threshold=-12.0)
    # error metric: lower RMSE is better; orient negates. test=10 (oriented -10) is the worse.
    # folds 10, 11, 20 -> oriented -10,-11,-20; only -20 < -12 -> n_below = 1
    comp = calc.compute(val_primary=9.0, test_primary=10.0, per_fold_test=[10.0, 11.0, 20.0])
    assert comp == pytest.approx(min(-9.0, -10.0) - 0.5 * 1)  # -10.5


def test_fingerprint_detects_metric_rewrite():
    """A changed metric definition changes the fingerprint and trips assert_fingerprint."""
    frozen = CompositeCalculator("lat_weighted_rmse", higher_is_better=False,
                                 penalty_weight=0.5, below_threshold=-12.0).fingerprint()

    # identical definition -> same fingerprint, no raise
    same = CompositeCalculator("lat_weighted_rmse", higher_is_better=False,
                               penalty_weight=0.5, below_threshold=-12.0)
    same.assert_fingerprint(frozen)  # must not raise

    # mid-project rewrite: penalty weight changed -> fingerprint differs -> raises
    rewritten = CompositeCalculator("lat_weighted_rmse", higher_is_better=False,
                                    penalty_weight=0.1, below_threshold=-12.0)
    assert rewritten.fingerprint() != frozen
    with pytest.raises(MetricFingerprintError):
        rewritten.assert_fingerprint(frozen)


# ----------------------------- splits ---------------------------------------

def test_walk_forward_no_overlap_and_future_test():
    X, y, regimes = load_dataset({"data": {"n_years": 6, "seed": 0}})
    folds = WalkForwardSplit(n_blocks=5, n_initial=2, gap=1).split(len(X))
    assert len(folds) >= 2
    validate_no_overlap(folds)
    for fa in folds:
        # test is strictly in the future of train+val
        assert fa.train_idx.max() < fa.test_idx.min()
        assert fa.val_idx.max() < fa.test_idx.min()


def test_super_fold_covers_all_regimes():
    X, y, regimes = load_dataset({"data": {"n_years": 6, "seed": 0, "n_regimes": 3}})
    folds = SuperFoldSplit(test_fraction=0.15, val_fraction=0.15).split(len(X), groups=regimes)
    assert len(folds) == 1
    validate_no_overlap(folds)
    import numpy as np
    test_regimes = set(np.unique(regimes[folds[0].test_idx]).tolist())
    assert test_regimes == set(np.unique(regimes).tolist())  # every regime is tested


# ----------------------------- mini autoresearch loop -----------------------

def _base_config(results_dir, backbone="mlp", primary_metric="lat_weighted_rmse",
                 penalty_weight=0.5, below_threshold=-12.0, description="exp",
                 bb_cfg=None):
    return {
        "paths": {"results_dir": str(results_dir)},
        "primary_metric": primary_metric,
        "backbone": backbone,
        "backbone_config": bb_cfg or {"hidden": 16, "epochs": 200, "lr": 0.05},
        "data": {"n_years": 6, "seed": 0, "n_lags": 7, "n_regimes": 3},
        "split": {"name": "walk_forward", "n_blocks": 5, "n_initial": 2, "gap": 1},
        "composite": {
            "higher_is_better": False,
            "penalty_weight": penalty_weight,
            "below_threshold": below_threshold,
        },
        "seed": 0,
        "description": description,
    }


def _author(results_dir, exp_num, description):
    """Write a gate-passing pre-run reasoning entry for `exp_num`."""
    ann = ReasoningAnnotationsFile(Path(results_dir) / "reasoning_annotations.json")
    e = _complete_entry(exp_num)
    e.diagnosis = e.diagnosis + f" (Experiment {exp_num}: {description}.)"
    ann.commit_pre_run(e)  # raises if it fails the gate


def test_mini_loop_champion_only_updates_on_improvement(tmp_path):
    """A 3-experiment loop: champion record updates only when composite improves."""
    results_dir = tmp_path / "results"

    # --- Exp 1: persistence baseline (the reference) ---
    _author(results_dir, 1, "persistence baseline")
    rec1 = run_experiment(_base_config(results_dir, backbone="persistence",
                                       description="persistence baseline"))
    best1 = json.loads((results_dir / "best_config.json").read_text(encoding="utf-8"))
    assert best1["experiment_num"] == 1  # first experiment is always champion
    assert rec1.status == "KEEP"

    # --- Exp 2: a deliberately weak MLP (few epochs) — should usually DISCARD ---
    _author(results_dir, 2, "under-trained MLP")
    rec2 = run_experiment(_base_config(
        results_dir, backbone="mlp", description="under-trained MLP",
        bb_cfg={"hidden": 4, "epochs": 5, "lr": 0.01},
    ))
    best2 = json.loads((results_dir / "best_config.json").read_text(encoding="utf-8"))
    if rec2.composite > rec1.composite:
        assert rec2.status == "KEEP" and best2["experiment_num"] == 2
    else:
        assert rec2.status == "DISCARD" and best2["experiment_num"] == 1

    # --- Exp 3: a well-trained MLP — record updates iff it actually improves ---
    _author(results_dir, 3, "well-trained MLP")
    champ_before = json.loads((results_dir / "best_config.json").read_text(encoding="utf-8"))
    best_before = champ_before["composite"]
    rec3 = run_experiment(_base_config(
        results_dir, backbone="mlp", description="well-trained MLP",
        bb_cfg={"hidden": 24, "epochs": 400, "lr": 0.05},
    ))
    champ_after = json.loads((results_dir / "best_config.json").read_text(encoding="utf-8"))
    if rec3.composite > best_before:
        assert rec3.status == "KEEP"
        assert champ_after["experiment_num"] == 3
        assert champ_after["composite"] == pytest.approx(rec3.composite)
    else:
        assert rec3.status == "DISCARD"
        assert champ_after["composite"] == pytest.approx(best_before)

    # invariant: champion composite is the max over all logged experiments
    logged = [json.loads(l) for l in
              (results_dir / "experiment_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert champ_after["composite"] == pytest.approx(max(r["composite"] for r in logged))

    # the loop wrote a self-contained crash-recovery checkpoint
    ckpt = (results_dir / "project_autoresearch_checkpoint.md").read_text(encoding="utf-8")
    assert "Crash-Recovery Checkpoint" in ckpt and "Current champion" in ckpt


def test_runner_refuses_shallow_entry_end_to_end(tmp_path):
    """If the on-disk entry is shallow, run_experiment raises (gate is wired into the runner)."""
    results_dir = tmp_path / "results"
    ann = ReasoningAnnotationsFile(results_dir / "reasoning_annotations.json")
    # write a shallow entry directly (bypassing commit_pre_run's validation)
    ann.save({"1": ReasoningEntry(experiment_num=1, diagnosis="short", citations="(x2025)",
                                  hypothesis="try", prediction="up").to_dict()})
    with pytest.raises(ValueError, match="pre-run reasoning failed the gate"):
        run_experiment(_base_config(results_dir, backbone="persistence"))
