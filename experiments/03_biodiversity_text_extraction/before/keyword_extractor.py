"""BEFORE: brittle keyword / regex extractor for species-interaction triples.

This is the *traditional* approach an environmental scientist might reach for
without an LLM: scan each comment for a small set of relation verbs
("pollinates", "feeds on", ...) and pair them with the two nearest known species
names. It is deliberately literal:

  * it only fires on the *explicit* verb forms of each relation;
  * it assumes "<subject> <verb> <object>" word order (active voice);
  * it has no notion of synonyms ("rely on", "visited by"), passive voice
    ("defoliated by", "is visited by"), or indirect phrasing
    ("canopy shelters", "tracks how many", "clustered on").

The result is high *precision* (when it fires it is usually right) but poor
*recall* on the varied, real-world phrasings in the citizen-science comments.
That recall gap is exactly what the AFTER structured extractor closes.

Run from the repo root so ``import common`` resolves:

    python -m experiments.03_biodiversity_text_extraction.before.keyword_extractor
"""
from __future__ import annotations

import re

from common import SPECIES, RELATIONS

__all__ = ["extract", "extract_all"]

# Explicit surface verbs for each relation. Intentionally narrow: only the
# canonical *active-voice* verb of each relation is recognised. No synonyms, no
# passive markers ("defoliated by", "visited by"), no indirect phrasing. This is
# the whole point of the BEFORE baseline: a keyword list a scientist jots down
# in an afternoon, which then misses most of the varied real-world phrasings.
_RELATION_VERBS: dict[str, list[str]] = {
    "pollinates": ["pollinates", "pollinate", "pollinating"],
    "feeds_on": ["feeds on", "feed on", "feeding on", "browsed", "browses"],
    "parasitizes": ["parasitizes", "parasitize", "parasitizing"],
    "competes_with": ["competes with", "compete with", "competing with"],
    "depends_on": ["depends on", "depend on", "depending on"],
}

# Species sorted longest-first so multi-word names match before sub-strings
# (e.g. "white-tailed deer" before "deer", "stingless bee" before "bee").
_SPECIES_BY_LEN = sorted(SPECIES, key=len, reverse=True)


def _find_species_spans(text: str) -> list[tuple[int, str]]:
    """Return [(char_offset, species)] for every known species mention, in order.

    Uses a non-overlapping longest-match scan so "white-tailed deer" is taken as
    one species rather than emitting a spurious "deer".
    """
    lowered = text.lower()
    spans: list[tuple[int, str]] = []
    taken = [False] * len(lowered)
    for sp in _SPECIES_BY_LEN:
        start = 0
        while True:
            idx = lowered.find(sp, start)
            if idx == -1:
                break
            if not any(taken[idx:idx + len(sp)]):
                spans.append((idx, sp))
                for k in range(idx, idx + len(sp)):
                    taken[k] = True
            start = idx + len(sp)
    spans.sort(key=lambda t: t[0])
    return spans


def extract(text: str) -> list[tuple[str, str, str]]:
    """Extract (subject, relation, object) triples from one comment.

    Brittle rule: for each explicit relation verb found, take the nearest known
    species *before* the verb as the subject and the nearest known species
    *after* the verb as the object (active-voice "S V O" assumption).
    """
    lowered = text.lower()
    spans = _find_species_spans(text)
    if len(spans) < 2:
        return []  # need at least two species for a directed interaction

    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for relation, verbs in _RELATION_VERBS.items():
        for verb in verbs:
            for m in re.finditer(re.escape(verb), lowered):
                v_start, v_end = m.start(), m.end()
                # nearest species ending before the verb -> subject
                subj = None
                for off, sp in spans:
                    if off + len(sp) <= v_start:
                        subj = sp  # keep latest one before verb
                # nearest species starting after the verb -> object
                obj = None
                for off, sp in spans:
                    if off >= v_end:
                        obj = sp
                        break
                if subj and obj and subj != obj:
                    triple = (subj, relation, obj)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
    return triples


def extract_all(comments: list[dict]) -> list[list[tuple[str, str, str]]]:
    """Run :func:`extract` over a list of comment dicts (``{"text": ...}``)."""
    return [extract(c["text"]) for c in comments]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from common import citizen_comments

    sample = citizen_comments(n=12, seed=0)
    print(f"Relations recognised: {RELATIONS}")
    for c in sample:
        got = extract(c["text"])
        print(f"\n[{c['id']}] {c['text']!r}")
        print(f"   gold : {c['interactions']}")
        print(f"   keyword: {got}")
