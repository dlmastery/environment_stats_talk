"""Fast tests for Exp05 — the flagship autoresearch loop.

Run from the repo root:
    python -m pytest experiments/05_autoresearch_climate/tests -q

These verify the load-bearing properties of the flagship demo without re-running
the full ~minute loop more than once:

1. The loop produces >= 5 logged experiments.
2. The champion composite is NON-DECREASING across the loop (the runner only ever
   replaces the champion on improvement).
3. A deliberately shallow pre-run entry RAISES at commit_pre_run (the gate works).
4. best_config.json's primary_metric is stable (the frozen fingerprint forbids a
   mid-loop metric swap), and the champion's composite equals the max over the log.

The loop is executed ONCE in a session-scoped fixture (it writes to the real
``results/`` dir, which the loop wipes at its own start, so the run is idempotent).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `import autoresearch_env`, `import common`, and the experiment package resolve.
_EXP_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_loop  # noqa: E402  (experiments/05_autoresearch_climate/run_loop.py)
from autoresearch_env import ReasoningEntry, ReasoningAnnotationsFile  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"


@pytest.fixture(scope="module")
def loop_result():
    """Run the full gated loop ONCE and return (progression, champ_exp, champ_composite)."""
    return run_loop.run_loop()


# --------------------------------------------------------------------------- #
def test_loop_produces_at_least_five_experiments(loop_result):
    progression, _, _ = loop_result
    log = RESULTS_DIR / "experiment_log.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) >= 5, f"expected >=5 logged experiments, got {len(rows)}"
    assert len(progression) == len(rows)
    # experiment numbers are 1..N contiguous
    assert [r["experiment_num"] for r in rows] == list(range(1, len(rows) + 1))


def test_champion_composite_non_decreasing(loop_result):
    progression, champ_exp, champ_composite = loop_result
    running = float("-inf")
    champ_curve = []
    for (_n, _bb, _desc, comp, _test, _status) in progression:
        running = max(running, comp)
        champ_curve.append(running)
    # the best-so-far staircase never goes down
    assert all(b >= a - 1e-12 for a, b in zip(champ_curve, champ_curve[1:]))
    # the recorded final champion matches the max composite in the log
    assert champ_composite == pytest.approx(max(p[3] for p in progression))
    assert champ_curve[-1] == pytest.approx(champ_composite)
    # a KEEP only happens when this experiment set a new best
    for i, (_n, _bb, _desc, comp, _test, status) in enumerate(progression):
        prior_best = max([p[3] for p in progression[:i]], default=float("-inf"))
        if status == "KEEP":
            assert comp > prior_best - 1e-12
        else:
            assert comp <= prior_best + 1e-12


def test_shallow_pre_run_entry_raises_at_gate(tmp_path):
    """A deliberately shallow pre-run entry must RAISE at commit_pre_run (gate works)."""
    ann = ReasoningAnnotationsFile(tmp_path / "reasoning_annotations.json")
    shallow = ReasoningEntry(
        experiment_num=1,
        diagnosis="too short",                 # below the 60-word floor
        citations="(ClimateLLM2025)",          # bare tag: no venue/title/year/note
        hypothesis="just try the mlp",         # no mechanism keyword, below floor
        prediction="it should be better",      # no numeric range
    )
    with pytest.raises(ValueError) as exc:
        ann.commit_pre_run(shallow)
    msg = str(exc.value)
    assert "failed validation" in msg
    # the gate surfaced the specific deficiencies
    assert "diagnosis" in msg
    assert "numeric range" in msg
    assert "citations" in msg


def test_best_config_primary_metric_stable(loop_result):
    """best_config.json reports a single, stable primary_metric; fingerprint is frozen."""
    _, champ_exp, champ_composite = loop_result
    best = json.loads((RESULTS_DIR / "best_config.json").read_text(encoding="utf-8"))
    assert best["experiment_num"] == champ_exp
    assert best["primary_metric"] == "rmse"
    assert best["composite"] == pytest.approx(champ_composite)

    log = RESULTS_DIR / "experiment_log.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    # one metric, one frozen fingerprint across the WHOLE loop (Goodhart guard)
    assert {r["primary_metric"] for r in rows} == {"rmse"}
    assert len({r["composite_fingerprint"] for r in rows}) == 1
    assert best["composite_fingerprint"] == rows[0]["composite_fingerprint"]


def test_all_pre_run_entries_passed_the_gate(loop_result):
    """Every authored entry on disk satisfies the full reasoning blob (post-run) gate."""
    from autoresearch_env import validate_reasoning_blob

    _ = loop_result  # ensure the loop has run and written the annotations
    data = json.loads((RESULTS_DIR / "reasoning_annotations.json").read_text(encoding="utf-8"))
    assert len(data) >= 5
    for key, entry_dict in data.items():
        violations = validate_reasoning_blob(ReasoningEntry.from_dict(entry_dict))
        assert violations == [], f"Exp{key} reasoning blob has violations: {violations}"
