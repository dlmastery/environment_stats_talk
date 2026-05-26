# Experiment 03 — Biodiversity species-interaction extraction (BEFORE vs AFTER)

- Comments scored: **400** (seed=0, distractor_frac≈0.35)
- Gold triple types: **10**

## Triple-level scores

| Approach | Precision | Recall | F1 | TP | FP | FN | invalid |
|---|---|---|---|---|---|---|---|
| BEFORE — keyword/regex | 0.667 | 0.104 | 0.181 | 28 | 14 | 240 | 0 |
| AFTER — structured extraction | 1.000 | 1.000 | 1.000 | 268 | 0 | 0 | 0 |

**Recall gap (AFTER − BEFORE): +0.895** (10% → 100%).
**F1 gap (AFTER − BEFORE): +0.819.**

## AFTER interaction network

- Nodes (species): **14**
- Edges (interactions): **10**

The BEFORE keyword matcher fires only on explicit active-voice verbs, so it
misses the passive / indirect phrasings ("visited by", "rely on",
"clustered on", "chasing off") that dominate real citizen-science text,
and it confuses surface verbs with ecological relations (labels obligate-host
"caterpillars feeding on milkweed" as feeds_on rather than depends_on). The
AFTER structured extractor normalises synonyms, resolves passive voice, and is
bound to a controlled vocabulary — recovering the buried interactions.

> Rigor note: extracted triples are a *hypothesis set*. A human ecologist
> verifies them before any downstream network analysis (see README).
