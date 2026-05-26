"""Reasoning-annotation schema + gates for the env-stats autoresearch loop.

This is a SELF-CONTAINED, lightweight re-implementation of the two hard gates
from the source FX/generalized framework
(``dlmastery/autoresearch`` -> ``generalized_ml_autoresearch/core/reasoning.py``),
adapted to environmental statistics. It deliberately does NOT import that repo;
it depends only on the Python standard library so it runs anywhere (CPU, no keys).

Two non-negotiable gates, enforced before any experiment may launch:

1. **Citation Rigor** — a citation must carry author(s) + year + venue + a title
   (quoted) OR an arXiv ID + a one-clause relevance note. Bare parenthetical tags
   like ``(ClimateLLM)`` are rejected. The env-stats venue/anchor vocabulary
   (arXiv, JGR, GMD, ERL, WAF, Climate Dynamics, Technometrics/Environmetrics,
   ICML/ICLR/NeurIPS) is recognized.

2. **Reasoning-Blob Completeness** — every field has a word-count floor (mirroring
   the source) and a required-keyword set: the hypothesis must be *mechanistic*
   ("mechanism" / "because" / "per <paper>"), and the prediction must contain a
   *numeric range* (e.g. "skill +0.05 to +0.12"). A verdict must say
   KEEP / DISCARD / NEAR-MISS and a learning must mark an axis open/closed.

The runner (``runner.py``) refuses to launch an experiment whose pre-run
annotation is missing or fails either gate.

Credit: methodology adapted from the user's own ``dlmastery/autoresearch``
(``generalized_ml_autoresearch``). Re-implemented here for env-stats so the talk
repo stays self-contained.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ReasoningEntry",
    "ReasoningAnnotationsFile",
    "validate_citation_rigor",
    "validate_reasoning_blob",
    "validate_pre_run_entry",
    "WORD_COUNT_FLOORS",
    "PLACEHOLDER_SENTINELS",
]

# -------------------- field spec (mirrors source Reasoning Blob Completeness) -----

REQUIRED_FIELDS = (
    "diagnosis",
    "citations",
    "hypothesis",
    "prediction",
    "verdict",
    "learning",
    "_manual",
)

# Word-count floors taken from the source framework's table. Kept identical so the
# rigor bar does not silently drop when porting to env-stats.
WORD_COUNT_FLOORS = {
    "diagnosis": 60,
    "citations_single": 40,
    "citations_multi": 80,
    "hypothesis": 50,
    "prediction": 25,
    "verdict": 30,
    "learning": 40,
}

REQUIRED_KEYWORDS = {
    # the hypothesis must be stated mechanistically
    "hypothesis": ("mechanism", "because", "per "),  # any one
    "verdict": ("KEEP", "DISCARD", "NEAR-MISS"),      # any one
    "learning": ("axis closed", "axis open", "next try"),  # any one
}

# placeholder tokens the runner inserts when a pre-run annotation is missing;
# they must never survive into a launched experiment.
PLACEHOLDER_SENTINELS = (
    "TODO-REWRITE",
    "(auto-backfilled)",
    "(no citation tag)",
    "(no explicit citation)",
    "see research_journal.md",
)


# -------------------- data class --------------------


@dataclass
class ReasoningEntry:
    """One experiment's full reasoning record — pre- and post-run fields."""

    experiment_num: int
    diagnosis: str = ""
    citations: str = ""
    hypothesis: str = ""
    prediction: str = ""
    verdict: str = ""
    learning: str = ""
    _manual: bool = True
    _needs_rewrite: bool = False
    authored_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# -------------------- citation validator --------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_RE = re.compile(r"arXiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)

# Env-stats + ML venue vocabulary. Includes the climate/earth-science journals and
# weather/forecasting outlets the audience trusts, plus the ML venues the foundation
# models publish in.
_VENUE_TOKENS = (
    # ML / AI venues
    "NeurIPS", "ICML", "ICLR", "AAAI", "CVPR", "ACL", "EMNLP", "KDD",
    "TMLR", "JMLR", "COLM", "IJCAI", "IEEE", "arXiv", "Nature", "Science",
    # environmental / earth-science / statistics venues
    "JGR", "GMD", "ERL", "WAF", "MWR", "QJRMS", "BAMS",
    "Climate Dynamics", "Journal of Climate", "Geophysical",
    "Technometrics", "Environmetrics", "JRSS", "Annals of Applied Statistics",
    "Remote Sensing of Environment", "Water Resources Research",
    "Econometrica", "JASA", "Neural Computation", "Atmospheric Environment",
    # established scientific publishers (for the classical-baseline textbooks
    # in sota_catalog_env.yaml: Forecasting:PP, Box-Jenkins, Coles EVT)
    "OTexts", "Springer", "Wiley",
)


def validate_citation_rigor(citations_text: str) -> list[str]:
    """Return a list of violations; an empty list means the citation passes.

    Rules (mirroring the source Citation-Rigor gate):
      1. Not a placeholder sentinel.
      2. Contains at least one 4-digit year.
      3. Contains at least one recognized venue token.
      4. Contains an arXiv ID OR a quoted title.
      5. Contains a relevance note (an em-dash/hyphen clause or a "because"/"per "/...
         keyword explaining WHY the paper is cited).
      6. Not a bare parenthetical tag like ``(ClimateLLM2025)``.
      7. Meets the single-/multi-paper word-count floor.
    """
    violations: list[str] = []
    text = (citations_text or "").strip()

    if not text:
        return ["citations field is empty"]

    for sentinel in PLACEHOLDER_SENTINELS:
        if sentinel.lower() in text.lower():
            violations.append(f"citations contains placeholder sentinel: {sentinel!r}")

    if not _YEAR_RE.search(text):
        violations.append("no year found (expected a 4-digit year like 2024, 2025)")

    if not any(v.lower() in text.lower() for v in _VENUE_TOKENS):
        violations.append(
            "no venue found (expected one of: arXiv, ICML, ICLR, NeurIPS, JGR, GMD, "
            "ERL, WAF, Climate Dynamics, Technometrics, Environmetrics, ...)"
        )

    has_arxiv = bool(_ARXIV_RE.search(text))
    has_title_quotes = any(q in text for q in ("'", '"', "‘", "“"))
    if not (has_arxiv or has_title_quotes):
        violations.append(
            "no arXiv ID and no quoted title — need a 'Title' in quotes OR (arXiv:XXXX.XXXXX)"
        )

    has_relevance = any(
        token in text.lower()
        for token in ("— ", " - ", "because", "motivates", "per ", "requires",
                      "suggests", "predicts", "informs")
    )
    if not has_relevance:
        violations.append(
            "no relevance note found — add a one-sentence note after the citation "
            "(e.g. '— motivates latitude-weighted RMSE as the headline skill metric')"
        )

    if re.fullmatch(r"\(\w+\d{4}\)", text):
        violations.append("parenthetical-only citation tag rejected — expand to a full reference")

    # multi-paper detection: papers separated by ';' + newline
    papers = [p.strip() for p in re.split(r";\s*\n", text) if p.strip()]
    is_multi = len(papers) >= 2
    word_count = len(text.split())
    floor = WORD_COUNT_FLOORS["citations_multi" if is_multi else "citations_single"]
    if word_count < floor:
        violations.append(
            f"citation text has {word_count} words; floor is {floor} "
            f"({'multi-paper' if is_multi else 'single paper'})"
        )

    return violations


# -------------------- reasoning-blob validator --------------------


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def _has_any_placeholder(text: str) -> bool:
    lower = (text or "").lower()
    return any(s.lower() in lower for s in PLACEHOLDER_SENTINELS)


# numeric-range matcher: "+0.05 to +0.12", "8.0-14.0", "0.6 to 0.7"
_RANGE_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?\s*(?:to|[-–])\s*[+\-]?\d+(?:\.\d+)?")


def validate_reasoning_blob(entry: ReasoningEntry) -> list[str]:
    """Full post-run validator. Returns a list of violations (empty == valid)."""
    violations: list[str] = []

    for f in REQUIRED_FIELDS:
        if not hasattr(entry, f):
            violations.append(f"field missing: {f}")

    for f in ("diagnosis", "citations", "hypothesis", "prediction", "verdict", "learning"):
        if _has_any_placeholder(getattr(entry, f, "")):
            violations.append(f"{f} contains placeholder sentinel — rewrite required")

    if _word_count(entry.diagnosis) < WORD_COUNT_FLOORS["diagnosis"]:
        violations.append(
            f"diagnosis has {_word_count(entry.diagnosis)} words; floor is {WORD_COUNT_FLOORS['diagnosis']}"
        )
    if _word_count(entry.hypothesis) < WORD_COUNT_FLOORS["hypothesis"]:
        violations.append(
            f"hypothesis has {_word_count(entry.hypothesis)} words; floor is {WORD_COUNT_FLOORS['hypothesis']}"
        )
    if _word_count(entry.prediction) < WORD_COUNT_FLOORS["prediction"]:
        violations.append(
            f"prediction has {_word_count(entry.prediction)} words; floor is {WORD_COUNT_FLOORS['prediction']}"
        )
    if _word_count(entry.verdict) < WORD_COUNT_FLOORS["verdict"]:
        violations.append(
            f"verdict has {_word_count(entry.verdict)} words; floor is {WORD_COUNT_FLOORS['verdict']}"
        )
    if _word_count(entry.learning) < WORD_COUNT_FLOORS["learning"]:
        violations.append(
            f"learning has {_word_count(entry.learning)} words; floor is {WORD_COUNT_FLOORS['learning']}"
        )

    if not any(kw.lower() in entry.hypothesis.lower() for kw in REQUIRED_KEYWORDS["hypothesis"]):
        violations.append(
            "hypothesis must include one of 'mechanism', 'because', or 'per [paper]' — "
            "state the change mechanistically"
        )
    if not any(kw.lower() in entry.verdict.lower() for kw in REQUIRED_KEYWORDS["verdict"]):
        violations.append("verdict must include KEEP, DISCARD, or NEAR-MISS")
    if not any(kw.lower() in entry.learning.lower() for kw in REQUIRED_KEYWORDS["learning"]):
        violations.append(
            "learning must include 'axis closed', 'axis open', or 'next try: ...' language"
        )

    violations.extend(f"citations: {v}" for v in validate_citation_rigor(entry.citations))

    if not _RANGE_RE.search(entry.prediction):
        violations.append(
            "prediction must contain a numeric range (e.g. 'skill +0.05 to +0.12'); "
            "single-point predictions are insufficient"
        )

    return violations


def validate_pre_run_entry(entry: ReasoningEntry) -> list[str]:
    """Pre-run gate: only diagnosis/citations/hypothesis/prediction are required.

    verdict and learning are authored AFTER the run, so they are not checked here.
    The runner calls this before launching an experiment.
    """
    violations: list[str] = []

    if _word_count(entry.diagnosis) < WORD_COUNT_FLOORS["diagnosis"]:
        violations.append(
            f"diagnosis has {_word_count(entry.diagnosis)} words; floor is {WORD_COUNT_FLOORS['diagnosis']}"
        )
    if _word_count(entry.hypothesis) < WORD_COUNT_FLOORS["hypothesis"]:
        violations.append(
            f"hypothesis has {_word_count(entry.hypothesis)} words; floor is {WORD_COUNT_FLOORS['hypothesis']}"
        )
    if _word_count(entry.prediction) < WORD_COUNT_FLOORS["prediction"]:
        violations.append(
            f"prediction has {_word_count(entry.prediction)} words; floor is {WORD_COUNT_FLOORS['prediction']}"
        )

    if not any(kw.lower() in entry.hypothesis.lower() for kw in REQUIRED_KEYWORDS["hypothesis"]):
        violations.append("hypothesis must include one of 'mechanism', 'because', or 'per [paper]'")
    if not _RANGE_RE.search(entry.prediction):
        violations.append("prediction must contain a numeric range")

    for f in ("diagnosis", "citations", "hypothesis", "prediction"):
        if _has_any_placeholder(getattr(entry, f, "")):
            violations.append(f"{f} contains placeholder sentinel")

    violations.extend(f"citations: {v}" for v in validate_citation_rigor(entry.citations))
    return violations


# -------------------- persistence --------------------


class ReasoningAnnotationsFile:
    """JSON file at ``<results_dir>/reasoning_annotations.json``.

    Keys are stringified experiment numbers; values are ReasoningEntry dicts.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def commit_pre_run(self, entry: ReasoningEntry) -> None:
        """Validate and persist a pre-run entry. Raises ValueError if it fails the gate."""
        violations = validate_pre_run_entry(entry)
        if violations:
            raise ValueError(
                "Pre-run reasoning entry failed validation. Fix these before launching:\n  - "
                + "\n  - ".join(violations)
            )
        data = self.load()
        data[str(entry.experiment_num)] = entry.to_dict()
        self.save(data)

    def commit_post_run(self, experiment_num: int, verdict: str, learning: str) -> list[str]:
        """Merge verdict/learning into an existing entry. Returns any validation violations."""
        data = self.load()
        key = str(experiment_num)
        if key not in data:
            data[key] = ReasoningEntry(
                experiment_num=experiment_num,
                diagnosis="TODO-REWRITE: pre-run annotation missing — author diagnosis.",
                citations="TODO-REWRITE: pre-run annotation missing — author citations.",
                hypothesis="TODO-REWRITE: pre-run annotation missing — author hypothesis.",
                prediction="TODO-REWRITE: pre-run annotation missing — author prediction.",
                verdict=verdict,
                learning=learning,
                _manual=False,
                _needs_rewrite=True,
            ).to_dict()
        else:
            data[key]["verdict"] = verdict
            data[key]["learning"] = learning
        self.save(data)
        return validate_reasoning_blob(ReasoningEntry.from_dict(data[key]))
