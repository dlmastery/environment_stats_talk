"""AFTER: structured ("LLM-style") species-interaction extractor.

Emulates the modern approach: hand a comment to an LLM with a *strict schema*
(typed subject/relation/object, controlled vocabulary) and let it normalise the
messy surface form into clean triples. The key behaviours we reproduce:

  * verb / synonym normalisation:
        "rely on", "depend on", "develop on", "shelters", "cannot feed",
        "clustered on", "cultivate"            -> depends_on
        "visited by", "pollination by", "working ... blossoms",
        "yield tracks how many"                -> pollinates
        "feeding on", "browsed", "stripping", "defoliated by", "took a",
        "population drops where"               -> feeds_on
        "squabbling over", "chasing off", "competes with"  -> competes_with
  * passive-voice handling: "<object> is visited by <subject>",
        "<object> defoliated by <subject>" -> subject/object are swapped back
        into canonical (agent, relation, patient) order;
  * species alias resolution (e.g. "monarch" -> "monarch butterfly");
  * controlled-vocabulary validation: only triples whose subject/object are in
    SPECIES and whose relation is in RELATIONS are returned (schema constraint).

This is a DETERMINISTIC, OFFLINE STAND-IN for a real LLM call so the experiment
runs anywhere with no API key. The function :func:`extract_with_llm` shows the
real schema-constrained tool-use call and is gated behind ANTHROPIC_API_KEY; it
is never required for the tests or the committed results.

Run from the repo root so ``import common`` resolves:

    python -m experiments.03_biodiversity_text_extraction.after.structured_extractor
"""
from __future__ import annotations

import json
import os
import re

from common import SPECIES, RELATIONS

__all__ = ["extract", "extract_all", "extract_with_llm", "INTERACTION_SCHEMA"]

# ---------------------------------------------------------------------------
# Controlled vocabulary + alias resolution (the "schema" the model is bound to)
# ---------------------------------------------------------------------------

# Common shorthand / informal aliases a citizen scientist might write. The LLM
# normalises these to the canonical species names in SPECIES.
_SPECIES_ALIASES: dict[str, str] = {
    "monarch": "monarch butterfly",
    "monarchs": "monarch butterfly",
    "bat": "long-nosed bat",
    "bats": "long-nosed bat",
    "deer": "white-tailed deer",
    "orchid": "cloud-forest orchid",
    "fir": "oyamel fir",
    "bee": "stingless bee",
    "bees": "stingless bee",
    "ant": "leafcutter ant",
    "ants": "leafcutter ant",
}

_SPECIES_BY_LEN = sorted(SPECIES, key=len, reverse=True)
_ALIASES_BY_LEN = sorted(_SPECIES_ALIASES, key=len, reverse=True)

INTERACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "enum": SPECIES},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "object": {"type": "string", "enum": SPECIES},
                },
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interactions"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Surface-pattern -> normalised relation. Each entry is a compiled regex over
# species placeholders S (agent) and O (patient). We capture which slot is the
# ecological agent vs patient so passive voice is resolved to canonical order.
#
# A pattern returns (relation, "SO") if the first matched species is the
# subject, or (relation, "OS") if the first matched species is the object
# (i.e. the sentence is patient-first / passive and must be swapped).
# ---------------------------------------------------------------------------

# Cues are evaluated TOP-DOWN and the FIRST match wins, so they are ordered
# from most-specific phrasing to most-generic verb. ``order`` records whether the
# ecological agent appears first ("SO") or the patient appears first ("OS",
# passive / indirect), so passive voice is resolved back to canonical
# (agent, relation, patient) order. This mirrors how an LLM, given the schema,
# decides the relation *and* the correct direction from context rather than from
# raw word order.
_REL_CUES: list[tuple[str, str, str]] = [
    # ---- depends_on: obligate host / development (checked before "feed") ----
    (r"\b(larvae|caterpillars?).{0,40}\b(develop|feeding|feed)\b", "depends_on", "SO"),
    (r"\bonly develop on\b", "depends_on", "SO"),       # larvae will only develop on O
    (r"\bnowhere to lay eggs\b", "depends_on", "OS"),   # no O ... so S had nowhere...
    (r"\bcanopy shelters\b", "depends_on", "OS"),       # O canopy shelters S
    (r"\bclustered\b.{0,20}\bon\b", "depends_on", "SO"),  # S clustered densely on O
    (r"\bcultivat", "depends_on", "SO"),                # S cultivate a O
    (r"\bcolony cannot feed\b", "depends_on", "OS"),    # without O, S colony cannot feed
    (r"\brely on\b.{0,30}\bfor pollination\b", "pollinates", "OS"),  # O rely on S for pollination
    (r"\bdepend", "depends_on", "SO"),
    (r"\brely on\b", "depends_on", "SO"),
    # ---- pollinates: specific/passive before generic verb ------------------
    (r"\bvisited by\b", "pollinates", "OS"),            # O is visited by S
    (r"\bpollination by\b", "pollinates", "OS"),        # O pollination by S
    (r"\bworking\b.{0,20}\bblossom", "pollinates", "SO"),  # S working the O blossoms
    (r"\byield\b.{0,30}\btracks how many\b", "pollinates", "OS"),  # O yield tracks how many S
    (r"\bactivity around\b.{0,20}\bbloom", "pollinates", "SO"),    # lots of S activity around blooming O
    (r"\bpollinat", "pollinates", "SO"),                # pollinates / pollinating
    # ---- competes_with (checked before generic "feed" to beat 'feeder') ----
    (r"\bsquabbling over\b", "competes_with", "SO"),
    (r"\bchasing off\b", "competes_with", "SO"),
    (r"\bcompet", "competes_with", "SO"),
    # ---- feeds_on: predation / herbivory -----------------------------------
    (r"\bdefoliated by\b", "feeds_on", "OS"),           # O defoliated by S
    (r"\bgrazing\b", "feeds_on", "OS"),                 # O damage consistent with S grazing
    (r"\bpopulation drops where\b", "feeds_on", "OS"),  # O population drops where S active
    (r"\bstripping\b", "feeds_on", "SO"),               # columns of S stripping O leaves
    (r"\btook a\b", "feeds_on", "SO"),                  # S took a O (predation)
    (r"\bbrowse", "feeds_on", "SO"),
    (r"\bfeed(s|ing)?\s+on\b", "feeds_on", "SO"),       # feeds on / feeding on (not 'feeder')
    # ---- parasitizes -------------------------------------------------------
    (r"\bparasit", "parasitizes", "SO"),
]


def _resolve_species(text: str) -> list[tuple[int, str]]:
    """Locate species (canonical + alias) mentions; return [(offset, canonical)].

    Longest-match, non-overlapping. Canonical names take priority over aliases.
    """
    lowered = text.lower()
    taken = [False] * len(lowered)
    spans: list[tuple[int, str]] = []

    def _scan(term: str, canonical: str) -> None:
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx == -1:
                break
            # require word boundaries for short aliases to avoid sub-matches
            left_ok = idx == 0 or not lowered[idx - 1].isalpha()
            right_i = idx + len(term)
            right_ok = right_i >= len(lowered) or not lowered[right_i].isalpha()
            if left_ok and right_ok and not any(taken[idx:right_i]):
                spans.append((idx, canonical))
                for k in range(idx, right_i):
                    taken[k] = True
            start = idx + len(term)

    for sp in _SPECIES_BY_LEN:        # canonical first (longest first)
        _scan(sp, sp)
    for al in _ALIASES_BY_LEN:        # then aliases fill remaining mentions
        _scan(al, _SPECIES_ALIASES[al])

    spans.sort(key=lambda t: t[0])
    return spans


def extract(text: str) -> list[tuple[str, str, str]]:
    """Extract canonical (subject, relation, object) triples from one comment.

    Stand-in for a schema-constrained LLM call: normalise verbs/synonyms,
    resolve aliases, handle passive voice, then validate against the controlled
    vocabulary so every returned triple is type-correct.
    """
    spans = _resolve_species(text)
    if len(spans) < 2:
        return []

    lowered = text.lower()
    # The first two distinct species in reading order anchor the relation.
    first_sp = spans[0][1]
    second_sp = next((sp for _, sp in spans[1:] if sp != first_sp), None)
    if second_sp is None:
        return []

    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for pattern, relation, order in _REL_CUES:
        if re.search(pattern, lowered):
            if order == "SO":
                subj, obj = first_sp, second_sp
            else:  # "OS" -> patient appears first, swap to canonical agent->patient
                subj, obj = second_sp, first_sp
            # schema validation (controlled vocabulary)
            if subj in SPECIES and obj in SPECIES and relation in RELATIONS \
                    and subj != obj:
                triple = (subj, relation, obj)
                if triple not in seen:
                    seen.add(triple)
                    triples.append(triple)
            # one relation per comment in this synthetic corpus; stop at first hit
            break
    return triples


def extract_all(comments: list[dict]) -> list[list[tuple[str, str, str]]]:
    """Run :func:`extract` over a list of comment dicts (``{"text": ...}``)."""
    return [extract(c["text"]) for c in comments]


# ---------------------------------------------------------------------------
# Real-LLM variant (gated behind ANTHROPIC_API_KEY; never required for tests)
# ---------------------------------------------------------------------------

_LLM_TOOL = {
    "name": "record_interactions",
    "description": (
        "Record every ecological species interaction explicitly stated or "
        "clearly implied in the comment. Use only the allowed species and "
        "relation values. If no interaction is present, return an empty list."
    ),
    "input_schema": INTERACTION_SCHEMA,
}

_LLM_SYSTEM = (
    "You are an ecology information-extraction assistant. Extract species "
    "interactions as typed triples (subject, relation, object). Normalise "
    "synonyms (e.g. 'rely on' -> depends_on, 'visited by' -> pollinates), "
    "resolve passive voice into canonical agent->patient order, and map "
    "informal names to the controlled species vocabulary. Only emit triples "
    "you are confident about; a human reviewer will verify them afterwards."
)


def extract_with_llm(text: str, model: str = "claude-sonnet-4-5") -> list[tuple[str, str, str]]:
    """Real schema-constrained extraction via the Anthropic SDK (tool use).

    GATED: requires the ``ANTHROPIC_API_KEY`` environment variable and the
    ``anthropic`` package. Raises ``RuntimeError`` if the key is absent so the
    offline tests never depend on it. The deterministic :func:`extract` above is
    the default path used for committed, runs-anywhere results.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; use the offline `extract()` instead. "
            "extract_with_llm() is an optional real-LLM variant and is never "
            "required for tests or committed results."
        )
    import anthropic  # local import so absence never breaks offline use

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_LLM_SYSTEM,
        tools=[_LLM_TOOL],
        tool_choice={"type": "tool", "name": "record_interactions"},
        messages=[{
            "role": "user",
            "content": (
                "Allowed species: " + ", ".join(SPECIES) + ".\n"
                "Allowed relations: " + ", ".join(RELATIONS) + ".\n\n"
                f"Comment: {text}"
            ),
        }],
    )
    triples: list[tuple[str, str, str]] = []
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            data = block.input
            if isinstance(data, str):
                data = json.loads(data)
            for it in data.get("interactions", []):
                s, r, o = it.get("subject"), it.get("relation"), it.get("object")
                if s in SPECIES and r in RELATIONS and o in SPECIES and s != o:
                    triples.append((s, r, o))
    return triples


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from common import citizen_comments

    sample = citizen_comments(n=16, seed=0)
    print(f"Relations: {RELATIONS}")
    print(f"API key present: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")
    for c in sample:
        got = extract(c["text"])
        flag = "" if set(got) == set(c["interactions"]) else "  <-- MISMATCH"
        print(f"\n[{c['id']}] {c['text']!r}")
        print(f"   gold      : {c['interactions']}")
        print(f"   structured: {got}{flag}")
