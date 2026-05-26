# Experiment 03 — Biodiversity species-interaction extraction from text (BEFORE vs AFTER)

**Task.** Read short citizen-science comments (think iNaturalist field notes) and
pull out the **species interactions** they describe as typed triples
`(subject, relation, object)` — e.g. `(hummingbird, pollinates, agave)`. The
relation vocabulary is fixed:

```
pollinates | feeds_on | parasitizes | competes_with | depends_on
```

Interactions are the raw material of food webs and pollination networks, yet most
of them live in unstructured prose, not in databases. The question for this
experiment: how much of that signal can you recover **before** (a keyword/regex
script you write in an afternoon) versus **after** (a schema-constrained,
LLM-style structured extractor)?

---

## The before/after story

| | BEFORE — keyword / regex | AFTER — structured extraction |
|---|---|---|
| Method | match a hand-listed set of active-voice verbs, pair with nearest species | normalise verbs/synonyms, resolve passive voice + aliases, bind to a typed schema |
| Phrasings it handles | only explicit `"X pollinates Y"`, `"feeding on"` | also `"visited by"`, `"rely on"`, `"clustered on"`, `"canopy shelters"`, `"chasing off"`, `"defoliated by"`, … |
| Passive voice | gets the direction backwards or misses it | resolves to canonical `agent → patient` |
| Relation vs verb | confuses them (`"caterpillars feeding on milkweed"` → `feeds_on`, but the ecological relation is `depends_on`) | distinguishes them from context |
| **Precision** | **0.667** (decent) | **1.000** |
| **Recall** | **0.104** (poor) | **1.000** |
| **F1** | **0.181** | **1.000** |
| Authoring effort | ~an afternoon to draft + endless rule patching as new phrasings appear | minutes to specify the schema + prompt; the model generalises |
| Annotation effort to label a 400-comment set | **hours** of manual triple tagging | **minutes** to extract a draft, then a human verifies |

*(Committed numbers from `results/metrics.json`, `n = 400`, `seed = 0`. The
recall gap — AFTER − BEFORE ≈ **+0.90** — is the headline finding.)*

### Why BEFORE under-recalls

The synthetic corpus mirrors real citizen-science text: ~35% of comments are
pure distractors (no interaction), and the interaction comments use deliberately
varied phrasing — passive (`"the agave is visited by the bat"`), indirect
(`"coffee yield tracks how many bees are around"`), and figurative
(`"hummingbirds chasing off bees"`). A literal verb list catches almost none of
these, so recall collapses to ~10% even though its precision stays decent.

### Why AFTER recovers them

The structured extractor emulates how you would actually use a modern LLM: give
it the **controlled vocabulary as a schema** and let it map messy surface forms
onto clean, typed triples — normalising synonyms (`"rely on"` → `depends_on`,
`"visited by"` → `pollinates`), resolving passive voice back to
`agent → patient` order, and rejecting anything outside the allowed species /
relation sets. On this synthetic benchmark it recovers every gold interaction.

> The 1.0 score is the *ceiling on a controlled synthetic corpus* where the
> vocabulary is closed. On real data, expect lower (and noisier) numbers — which
> is exactly why the human-verification step below is mandatory.

---

## Files

```
03_biodiversity_text_extraction/
├── before/keyword_extractor.py      # brittle active-voice verb + nearest-species matcher
├── after/structured_extractor.py    # offline schema-constrained extractor + real-LLM variant
├── run_before_after.py              # scores both, builds the network, writes results/
├── tests/test_exp03.py              # fast offline pytest (<30s, no network)
└── results/
    ├── metrics.json                 # triple-level P/R/F1 + graph + triples
    ├── before_after_bars.png        # precision/recall/F1 bars
    ├── interaction_network.png      # AFTER species-interaction graph (edges by relation)
    └── summary.md                   # human-readable scorecard
```

---

## How to run

Run from the **repo root** so `import common` resolves.

```bash
# full committed run (n = 400)
python experiments/03_biodiversity_text_extraction/run_before_after.py

# larger / custom corpus
python experiments/03_biodiversity_text_extraction/run_before_after.py --n 800 --seed 1

# fast smoke run (n = 60), still writes every artifact
python experiments/03_biodiversity_text_extraction/run_before_after.py --quick

# tests (fast, offline, no API key)
python -m pytest experiments/03_biodiversity_text_extraction/tests -q
```

Everything above is **runs-anywhere**: deterministic synthetic data, CPU only,
**no API key**.

---

## The real-LLM variant (optional)

`after/structured_extractor.py` ships `extract_with_llm(text)`, a *real*
schema-constrained extraction call using the Anthropic SDK with tool use: the
controlled vocabulary is encoded as a JSON-schema `input_schema` and the model is
forced to emit only valid `(subject, relation, object)` triples.

It is **gated behind the `ANTHROPIC_API_KEY` environment variable** — if the key
is absent it raises immediately, so the offline `extract()` (the deterministic
stand-in) is what the tests and committed results use. To try the real call:

```bash
export ANTHROPIC_API_KEY=sk-...        # Windows: $env:ANTHROPIC_API_KEY = "sk-..."
python -c "import sys; sys.path.insert(0,'experiments/03_biodiversity_text_extraction/after'); \
import structured_extractor as s; print(s.extract_with_llm('the agave is visited by the long-nosed bat at night'))"
```

The deterministic `extract()` is intentionally a faithful *behavioural* stand-in
(same schema, same normalisation rules) so the headline before/after gap is
reproducible without a key or network.

---

## Swapping in real data (iNaturalist / GBIF)

The synthetic generator (`common/synthetic_biodiversity.py`) emits records with
the schema `{"id", "text", "interactions"}`. To run on real text:

1. Pull observation **comments / descriptions / field notes** from
   **iNaturalist** (per-observation `description` and comment fields) or
   occurrence remarks from **GBIF** (`occurrenceRemarks`, `eventRemarks`).
2. Map each record to `{"id": <obs id>, "text": <comment>, "interactions": []}`
   (leave `interactions` empty for unlabeled data; fill it for a held-out gold
   set).
3. Point `run_before_after.py` at your loader instead of `citizen_comments(...)`
   — nothing else changes; the scorer and graph builder are data-agnostic.

Species and relation vocabularies (`SPECIES`, `RELATIONS`) would be replaced with
a domain taxonomy (e.g. a GBIF backbone name list) for the real run.

---

## Rigor: a human verifies before analysis

Extracted triples are a **hypothesis set, not ground truth.** Both the keyword
and the structured extractor can misread phrasing, conflate a verb with a
relation, or hallucinate an interaction that the text only hints at. Before any
triple feeds a downstream food-web / pollination-network analysis, a human
ecologist reviews it — confirming the relation type and direction, and rejecting
spurious or low-confidence extractions. The LLM accelerates the first pass from
hours of manual annotation to minutes; the scientist keeps final say on what
enters the network. This division of labour (machine drafts, human validates) is
the same statistical-rigor discipline applied across this repo.

*Context (generic):* using LLMs to extract species interactions and ecological
relations from unstructured text — literature, field notes, and citizen-science
comments — is an active and growing line of work in computational ecology. We
describe the approach generically here and do not attribute specific
papers, datasets, or quantitative results.
