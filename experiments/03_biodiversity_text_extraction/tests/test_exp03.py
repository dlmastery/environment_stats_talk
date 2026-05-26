"""Fast offline tests for Exp03 (no network, no API key, < 30 s).

Asserts the BEFORE-vs-AFTER story holds:
  * BEFORE (keyword/regex) has HIGH precision but POOR recall;
  * BEFORE recall < AFTER recall (the recall gap);
  * AFTER F1 > BEFORE F1;
  * every extracted triple is schema-valid (subject/object in SPECIES,
    relation in RELATIONS);
  * the AFTER interaction graph has > 0 nodes and > 0 edges.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# --- resolve paths and import the experiment modules by file path ----------
# The experiment dir name ("03_...") is not a valid Python package identifier,
# so we load the before/after extractor modules directly from their files and
# put the repo root on sys.path so they can ``import common``.
_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_REPO_ROOT = _EXP.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


before_mod = _load("exp03_keyword_extractor", _EXP / "before" / "keyword_extractor.py")
after_mod = _load("exp03_structured_extractor", _EXP / "after" / "structured_extractor.py")
runner = _load("exp03_runner", _EXP / "run_before_after.py")

from common import citizen_comments, SPECIES, RELATIONS  # noqa: E402

N = 400
SEED = 0


@pytest.fixture(scope="module")
def comments():
    return citizen_comments(n=N, seed=SEED)


@pytest.fixture(scope="module")
def before_scores(comments):
    return runner.score_extractor(before_mod.extract, comments)


@pytest.fixture(scope="module")
def after_scores(comments):
    return runner.score_extractor(after_mod.extract, comments)


# ---------------------------------------------------------------------------
# Validity of extracted triples (controlled vocabulary / schema)
# ---------------------------------------------------------------------------

def test_before_triples_are_schema_valid(comments):
    for c in comments:
        for s, r, o in before_mod.extract(c["text"]):
            assert s in SPECIES, s
            assert o in SPECIES, o
            assert r in RELATIONS, r


def test_after_triples_are_schema_valid(comments):
    for c in comments:
        for s, r, o in after_mod.extract(c["text"]):
            assert s in SPECIES, s
            assert o in SPECIES, o
            assert r in RELATIONS, r


def test_no_invalid_triples_reported(before_scores, after_scores):
    assert before_scores["invalid_triples"] == 0
    assert after_scores["invalid_triples"] == 0


# ---------------------------------------------------------------------------
# The BEFORE/AFTER story
# ---------------------------------------------------------------------------

def test_before_has_high_precision(before_scores):
    # brittle but literal: when it fires it is usually right
    assert before_scores["precision"] >= 0.6


def test_before_has_poor_recall(before_scores):
    assert before_scores["recall"] < 0.3


def test_recall_gap_after_beats_before(before_scores, after_scores):
    assert after_scores["recall"] > before_scores["recall"]
    # the gap is the headline finding — require it to be substantial
    assert (after_scores["recall"] - before_scores["recall"]) > 0.3


def test_after_f1_beats_before_f1(before_scores, after_scores):
    assert after_scores["f1"] > before_scores["f1"]


def test_after_precision_not_degraded(after_scores):
    # structured extraction must not trade away precision for recall
    assert after_scores["precision"] >= 0.8


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def test_after_graph_has_nodes_and_edges(comments):
    triples = runner.collect_triples(after_mod.extract, comments)
    graph = runner.build_graph(triples)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0


# ---------------------------------------------------------------------------
# Real-LLM variant is gated and never required
# ---------------------------------------------------------------------------

def test_extract_with_llm_gated_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        after_mod.extract_with_llm("a hummingbird pollinates the agave")


# ---------------------------------------------------------------------------
# End-to-end driver smoke (quick mode, writes artifacts to a temp dir)
# ---------------------------------------------------------------------------

def test_driver_quick_writes_artifacts(tmp_path):
    metrics = runner.main(["--quick", "--outdir", str(tmp_path)])
    for name in ("metrics.json", "before_after_bars.png",
                 "interaction_network.png", "summary.md"):
        assert (tmp_path / name).exists(), name
    assert metrics["after"]["f1"] > metrics["before"]["f1"]
    assert metrics["after_graph"]["nodes"] > 0
    assert metrics["after_graph"]["edges"] > 0
