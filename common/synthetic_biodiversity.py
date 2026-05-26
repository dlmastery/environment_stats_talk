"""Synthetic biodiversity / citizen-science text generator.

Produces short "citizen-science comments" (think iNaturalist notes) that embed
species *interactions*, plus a gold-standard label set. This lets Exp03 measure
extraction quality (precision/recall) of a BEFORE approach (keyword/regex) vs an
AFTER approach (structured-LLM extraction).

The phrasing is deliberately varied (synonyms, passive voice, indirection) so a
brittle keyword matcher under-recalls — mirroring the real-world finding that
LLMs unlock interactions buried in unstructured text.

Swap-in note: replace `citizen_comments` with real iNaturalist/GBIF comment text;
keep the dict schema {"id", "text", "interactions"} and Exp03 works unchanged.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SPECIES", "RELATIONS", "citizen_comments", "gold_interactions"]

# Latin-America-relevant species vocabulary (common names used in the text).
SPECIES = [
    "monarch butterfly", "milkweed", "oyamel fir", "blue oak", "hummingbird",
    "agave", "long-nosed bat", "coffee shrub", "stingless bee", "jaguar",
    "white-tailed deer", "cloud-forest orchid", "leafcutter ant", "fungus",
]

# Interaction (relation) types we want extracted.
RELATIONS = ["pollinates", "feeds_on", "parasitizes", "competes_with", "depends_on"]

# Ground-truth interaction templates: (subject, relation, object, [phrasings])
# Phrasings include easy ("X pollinates Y") and hard (passive / indirect) variants.
_TEMPLATES = [
    ("hummingbird", "pollinates", "agave",
     ["I watched a {s} {r_v} the {o} flowers at dawn.",
      "The {o} here seems to rely on the {s} for pollination.",
      "Lots of {s} activity around blooming {o} this week."]),
    ("long-nosed bat", "pollinates", "agave",
     ["At night the {o} is visited by the {s}.",
      "{o} pollination by the {s} looked very active."]),
    ("monarch butterfly", "depends_on", "milkweed",
     ["The {s} larvae will only develop on {o}.",
      "No {o} patches left, so the {s} had nowhere to lay eggs.",
      "{s} caterpillars were feeding on the {o} leaves."]),
    ("monarch butterfly", "depends_on", "oyamel fir",
     ["Overwintering {s} clustered densely on the {o}.",
      "The {o} canopy shelters the migrating {s}."]),
    ("stingless bee", "pollinates", "coffee shrub",
     ["Saw {s} working the {o} blossoms all morning.",
      "{o} yield here tracks how many {s} are around."]),
    ("leafcutter ant", "feeds_on", "blue oak",
     ["Columns of {s} were stripping the {o} leaves.",
      "The {o} sapling was defoliated by {s}."]),
    ("leafcutter ant", "depends_on", "fungus",
     ["The {s} cultivate a {o} in their nest.",
      "Without the {o}, the {s} colony cannot feed."]),
    ("white-tailed deer", "feeds_on", "cloud-forest orchid",
     ["A {s} had browsed the {o} right down to the stem.",
      "Found {o} damage consistent with {s} grazing."]),
    ("jaguar", "feeds_on", "white-tailed deer",
     ["Tracks suggest a {s} took a {o} near the river.",
      "The {o} population drops where {s} are active."]),
    ("hummingbird", "competes_with", "stingless bee",
     ["{s} and {o} were squabbling over the same nectar source.",
      "Noticed {s} chasing off {o} at the feeder."]),
]

# Distractor sentences (no interaction) to add noise / test false positives.
_DISTRACTORS = [
    "Beautiful sunny morning on the trail today.",
    "The river was running high after the rain.",
    "Took some photos of the {x} but it flew off quickly.",
    "Soil here is very dry compared to last season.",
    "Saw a single {x} resting on a rock, no other activity.",
    "Trail was muddy near the {x} grove.",
]


def gold_interactions() -> set[tuple[str, str, str]]:
    """The complete set of true (subject, relation, object) triples."""
    return {(s, r, o) for (s, r, o, _ph) in _TEMPLATES}


def citizen_comments(n: int = 200, seed: int = 0, distractor_frac: float = 0.35) -> list[dict]:
    """Generate ``n`` synthetic comments.

    Each record: ``{"id": int, "text": str, "interactions": [(s, r, o), ...]}``.
    ``distractor_frac`` of comments contain no interaction (label = empty list).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for i in range(n):
        if rng.random() < distractor_frac:
            tmpl = _DISTRACTORS[rng.integers(len(_DISTRACTORS))]
            text = tmpl.format(x=SPECIES[rng.integers(len(SPECIES))])
            records.append({"id": i, "text": text, "interactions": []})
            continue
        s, r, o, phrasings = _TEMPLATES[rng.integers(len(_TEMPLATES))]
        phr = phrasings[rng.integers(len(phrasings))]
        r_v = r.replace("_", " ")  # "feeds_on" -> "feeds on"
        text = phr.format(s=s, o=o, r_v=r_v)
        # Occasionally append a distractor clause to make extraction harder.
        if rng.random() < 0.3:
            text += " " + _DISTRACTORS[rng.integers(len(_DISTRACTORS))].format(
                x=SPECIES[rng.integers(len(SPECIES))])
        records.append({"id": i, "text": text, "interactions": [(s, r, o)]})
    return records
