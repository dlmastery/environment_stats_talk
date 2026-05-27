# Tutorial — Driving the Env-Stats AutoResearch Loop Yourself

**Audience:** an environmental statistician who wants to *run* the flagship loop,
not just read about it. By the end you will have authored a gate-passing pre-run
reasoning entry, launched one experiment, read the artifacts it produced, recovered
a session from a simulated crash, and understood exactly how the CPU smoke run scales
to the RTX 4090.

**Who is the researcher?** You — with **Claude Code as the outer loop**. There is no
separate Python "agent" that invents experiments. Claude Code (or you) diagnoses the
current champion, cites a real paper, states a mechanistic hypothesis and a numeric
prediction, runs **one** experiment through a gated runner, then writes a verdict and
the next diagnosis. The runner only *enforces* discipline; it never invents reasoning.

**Source credit.** The 7-step loop, the two gates, the
`min(val, test) − penalty·n_below_threshold` composite, the frozen fingerprint, the
winner archive, and the crash-recovery checkpoint are adapted from the user's own
`dlmastery/autoresearch` (`generalized_ml_autoresearch`). This repo's contribution is
the environmental-statistics specialisation in
[`autoresearch_env/`](../autoresearch_env/README.md).

**What you need:** Python 3.12, this repo, no API keys, no GPU. The whole reference
loop (Exp05) runs on CPU in **~38 seconds of compute**.

---

## 0. Orientation — the files you will touch

| File | What it is | Who writes it |
|---|---|---|
| `autoresearch_env/reasoning.py` | The two gates + `ReasoningEntry` / `ReasoningAnnotationsFile` | code (you read it) |
| `autoresearch_env/runner.py` | `run_experiment(config)` — one experiment per call | code |
| `autoresearch_env/sota_catalog_env.yaml` | Vetted, real citations that seed the citation gate | code |
| `experiments/05_autoresearch_climate/run_loop.py` | The reference six-experiment loop, fully authored | the model, as researcher |
| `.../results/experiment_log.jsonl` | One JSON row per experiment | runner |
| `.../results/best_config.json` | The current champion record (+ frozen fingerprint) | runner (on improvement) |
| `.../results/reasoning_annotations.json` | All reasoning blobs (pre + post-run) | you, via the annotations API |
| `.../results/project_autoresearch_checkpoint.md` | Self-contained crash-recovery checkpoint | runner (every experiment) |

Run the reference loop once so you have artifacts to inspect:

```bash
# from the repo root
python experiments/05_autoresearch_climate/run_loop.py
python -m pytest experiments/05_autoresearch_climate/tests -q
```

---

## 1. The 7-step loop

Each iteration runs the same seven steps. Steps **2–4 are machine-checked** before
step 5 is allowed to run — that is the whole point.

| # | Step | What you do | Where it lives |
|---|------|-------------|----------------|
| 1 | **Diagnose** | Read `experiment_log.jsonl` + the checkpoint; find the champion's weakest fold / year / climate regime. | you, reading artifacts |
| 2 | **Cite** | Pick a *real* reference from `sota_catalog_env.yaml` (or `CLAUDE.md` §6) that addresses that weakness. | `ReasoningEntry.citations` |
| 3 | **Hypothesize** | State the change *mechanistically* ("…because the mechanism is…"). | `ReasoningEntry.hypothesis` |
| 4 | **Predict** | Commit a *numeric range* for the expected change. | `ReasoningEntry.prediction` |
| 5 | **Execute** | Author the entry (gate runs here, **raises** on failure), then run **exactly one** experiment. | `commit_pre_run` → `run_experiment` |
| 6 | **Analyze** | Compare the realized composite to your prediction; write a verdict (KEEP / DISCARD / NEAR-MISS) + a learning. | `commit_post_run` |
| 7 | **Checkpoint** | The runner already rewrote the crash-recovery markdown — verify it, then loop back to step 1. | runner (automatic) |

One change per experiment, so every diff is attributable to a single cause.

---

## 2. Authoring a pre-run reasoning entry that PASSES the gates

Before any experiment launches, the runner calls `validate_pre_run_entry(entry)` and
**refuses to run** if it returns any violations. The pre-run gate checks four fields:
`diagnosis`, `citations`, `hypothesis`, `prediction`. (`verdict` and `learning` are
authored *after* the run and checked by the full-blob gate then.)

### 2a. Field requirements (exact, from `reasoning.py`)

| Field | Floor (words) | Required content |
|---|---|---|
| `diagnosis` | **≥ 60** | Plain prose: what the champion's weakness is (or scope-setting for Exp1). No keyword required. No placeholder sentinels. |
| `citations` | **≥ 40** single paper / **≥ 80** multi-paper | A 4-digit **year**; a recognised **venue** token; an **arXiv ID** *or* a quoted **'title'**; and a **relevance clause**. |
| `hypothesis` | **≥ 50** | Must be **mechanistic** — contain one of `mechanism`, `because`, or `per ` (note the trailing space, as in "per Box-Jenkins"). |
| `prediction` | **≥ 25** | Must contain a **numeric range**, e.g. `+0.05 to +0.12`, `1.79 to 1.84`, or `-1.84 to -1.80`. Single-point predictions are rejected. |

Post-run (the full-blob gate, via `commit_post_run`):

| Field | Floor (words) | Required content |
|---|---|---|
| `verdict` | **≥ 30** | Must contain one of `KEEP`, `DISCARD`, `NEAR-MISS`. |
| `learning` | **≥ 40** | Must contain one of `axis open`, `axis closed`, or `next try`. |

### 2b. The Citation-Rigor gate, in detail

`validate_citation_rigor` (in `reasoning.py`) rejects a citation unless ALL of:

1. **Not** a placeholder sentinel (`TODO-REWRITE`, `(auto-backfilled)`, `(no citation tag)`, `(no explicit citation)`, `see research_journal.md`).
2. Contains a **4-digit year** (`19xx`/`20xx`).
3. Contains a recognised **venue token**. The vocabulary includes ML venues
   (NeurIPS, ICML, ICLR, TMLR, JMLR, arXiv, Nature, Science, …), env/earth-science
   venues (JGR, GMD, ERL, WAF, MWR, QJRMS, BAMS, *Climate Dynamics*, *Journal of
   Climate*, *Technometrics*, *Environmetrics*, *Atmospheric Environment*, *Neural
   Computation*, *Water Resources Research*, …), and the publishers behind the
   classical-baseline textbooks (**OTexts**, **Springer**, **Wiley**).
4. Contains an **arXiv ID** (`arXiv:2310.10688`) **OR** a quoted **'title'**.
5. Contains a **relevance note** — a clause introduced by an em-dash/hyphen, or one
   of `because`, `motivates`, `per `, `requires`, `suggests`, `predicts`, `informs`.
6. Is **not** a bare parenthetical tag like `(ClimateLLM2025)`.
7. Meets the **40-word** floor (single paper) / **80-word** floor (multi-paper;
   papers are detected by being separated with `;` then a newline).

**This is why bare tags fail.** `(ClimateLLM2025)` has no venue spelled out, no
relevance clause, no quoted title, and far fewer than 40 words — it trips rules 3, 4,
5, 6, and 7 simultaneously.

### 2c. A worked example entry that PASSES (Exp2 — the first learned model)

This is the real Exp2 entry from `run_loop.py`. Copy the *shape*, not the literal text,
when you author your own.

```python
from autoresearch_env import ReasoningEntry, ReasoningAnnotationsFile

ann = ReasoningAnnotationsFile("experiments/05_autoresearch_climate/results/reasoning_annotations.json")

entry = ReasoningEntry(
    experiment_num=2,
    diagnosis=(
        "The Exp1 persistence champion ignores two things it structurally cannot represent: the "
        "curved shape of the annual cycle (it carries forward a single recent value rather than "
        "anticipating the seasonal turn) and the slow warming trend that makes 'yesterday' a "
        "slightly cold-biased estimate of 'today' on average. Its per-fold RMSE also drifts upward "
        "on the later, warmer test blocks, which is the signature of an unmodelled trend. The "
        "weakest aspect to attack first is therefore the missing seasonal-shape and trend signal, "
        "which a minimal learned model on lag and calendar features can supply cheaply."
    ),  # >= 60 words
    citations=(
        "Gardner & Dorling 1998 Atmospheric Environment 'Artificial neural networks (the "
        "multilayer perceptron) — a review of applications in the atmospheric sciences' — "
        "establishes the 1-hidden-layer MLP on lagged and seasonal inputs as the simplest neural "
        "baseline in the atmospheric sciences, and motivates this experiment because it predicts "
        "that a small perceptron on autoregressive lags plus a day-of-year encoding can capture "
        "the seasonal shape that a naive carry-forward misses, at negligible compute cost."
    ),  # year=1998, venue='Atmospheric Environment', quoted title, '— ... motivates ... because', >= 40 words
    hypothesis=(
        "A tiny one-hidden-layer MLP on 7 autoregressive lags plus a sin/cos day-of-year encoding "
        "will beat persistence by a small but real margin, because the mechanism is that the "
        "seasonal encoding lets the net anticipate the annual turn while the lag block recovers "
        "the AR(1) carry, so it corrects the seasonal-shape and warm-trend bias that persistence "
        "cannot. The gain should be modest precisely because persistence is already near-optimal "
        "for the dominant one-step autocorrelation, leaving only the structured residual to learn."
    ),  # contains 'because' AND 'mechanism', >= 50 words
    prediction=(
        "Aggregated walk-forward test RMSE should improve from the persistence floor by roughly "
        "0.05 to 0.15 deg C, landing in the range 1.80 to 1.92 deg C, with skill-vs-persistence "
        "in the range +0.02 to +0.08; the composite should rise from about -1.96 into the range "
        "-1.90 to -1.84, still with zero folds below the acceptance threshold."
    ),  # multiple numeric ranges ('0.05 to 0.15', '1.80 to 1.92', '+0.02 to +0.08'), >= 25 words
)

ann.commit_pre_run(entry)   # <-- GATE RUNS HERE. Raises ValueError if anything above fails.
```

`commit_pre_run` either persists the entry (gate passed) or raises a `ValueError`
listing every violation. There is no way to "skip" it — `run_experiment` re-runs the
same pre-run gate and refuses to launch if the entry is missing or shallow.

**What a FAIL looks like.** If you shrank the citation to `(GardnerDorling1998)` and
the prediction to `RMSE will drop a bit`, `commit_pre_run` would raise something like:

```
Pre-run reasoning entry failed validation. Fix these before launching:
  - prediction has 5 words; floor is 25
  - prediction must contain a numeric range
  - citations: no venue found (expected one of: arXiv, ICML, ICLR, NeurIPS, JGR, ...)
  - citations: no arXiv ID and no quoted title — need a 'Title' in quotes OR (arXiv:XXXX.XXXXX)
  - citations: no relevance note found — add a one-sentence note after the citation
  - citations: parenthetical-only citation tag rejected — expand to a full reference
  - citations: citation text has 1 words; floor is 40 (single paper)
```

The reference loop's test `tests/test_exp05.py::test_shallow_pre_run_entry_raises_at_gate`
asserts this bite: a shallow entry **must** raise at the gate.

---

## 3. Running one experiment

A config is a plain dict (or a YAML/JSON file). The exact shape the runner expects:

```yaml
paths: { results_dir: "experiments/05_autoresearch_climate/results" }
primary_metric: rmse                # or lat_weighted_rmse / acc / skill_vs_persistence / mae
backbone: mlp                       # or persistence  (only two backbones exist in the CPU package)
backbone_config: { hidden: 16, epochs: 400, lr: 0.05, weight_decay: 1.0e-4 }
data: { n_years: 14, seed: 0, n_lags: 7, n_regimes: 3 }
split: { name: walk_forward, n_blocks: 7, n_initial: 3, gap: 1 }   # or super_fold
composite: { higher_is_better: false, penalty_weight: 0.5, below_threshold: -12.0 }
seed: 0
description: "Exp2 — tiny MLP, 7 lags + seasonal encoding"
```

### Programmatically (the way the reference loop does it)

```python
from autoresearch_env import run_experiment

config = {
    "paths": {"results_dir": "experiments/05_autoresearch_climate/results"},
    "primary_metric": "rmse",
    "backbone": "mlp",
    "backbone_config": {"hidden": 16, "epochs": 400, "lr": 0.05, "weight_decay": 1e-4},
    "data": {"n_years": 14, "seed": 0, "n_lags": 7, "n_regimes": 3},
    "split": {"name": "walk_forward", "n_blocks": 7, "n_initial": 3, "gap": 1},
    "composite": {"higher_is_better": False, "penalty_weight": 0.5, "below_threshold": -12.0},
    "seed": 0,
    "description": "Exp2 — tiny MLP, 7 lags + seasonal encoding",
}

record = run_experiment(config)   # gate must already be satisfied for this exp_num
print(record.status, record.composite, record.test_primary)
```

### From the CLI

```bash
python -m autoresearch_env.runner --config path/to/exp.yaml --description "MLP, 7 lags"
```

The runner, in order: (b) checks the pre-run reasoning gate → (c) **asserts the frozen
fingerprint** → loads data, builds folds, asserts no train/val/test overlap → trains
the backbone per fold → computes the composite → appends a row to
`experiment_log.jsonl` → writes `best_config.json` **only if** the composite improved →
rewrites the checkpoint. Then *you* author the verdict + learning:

```python
from autoresearch_env import ReasoningAnnotationsFile
ann = ReasoningAnnotationsFile("experiments/05_autoresearch_climate/results/reasoning_annotations.json")
violations = ann.commit_post_run(
    experiment_num=2,
    verdict="KEEP. The MLP landed at test=1.8435 deg C, composite=-1.8503, a 0.1083 composite move "
            "versus the persistence floor; the gain is real but modest, as predicted.",   # >= 30 words, says KEEP
    learning="A learned baseline beats persistence but only slightly — honest modest delta. Axis open on "
             "feature design: the 7-lag window may be longer than the true memory horizon. Next try: shrink "
             "to 3 lags to cut redundant inputs and tighten out-of-sample extrapolation.",  # >= 40 words, 'axis open' + 'next try'
)
assert not violations, violations
```

---

## 4. Reading the artifacts

### `experiment_log.jsonl` — one JSON row per experiment (append-only)

The real Exp1 row contains (abridged): `experiment_num`, `backbone: "persistence"`,
`composite: -1.9586…`, `val_primary`, `test_primary: 1.9477…`,
`per_fold_test: [1.7733, 1.8521, 2.0271, 2.1382]`, `status: "KEEP"`,
`composite_fingerprint: "942df6a8450d8d57"`, `primary_metric: "rmse"`, plus
`per_fold_reports` (per fold: `fold_id`, `regime`, `n_test`, `rmse`, `mae`, `acc`) and
the full `config`. The four per-fold RMSE values climbing from 1.77 → 2.14 across
walk-forward blocks 3→6 *is* the unmodelled-warming-trend signature that the Exp2
diagnosis attacks — that is how you "diagnose" from this file.

To tabulate quickly:

```bash
python -c "import json,sys; [print(json.loads(l)['experiment_num'], json.loads(l)['backbone'], round(json.loads(l)['composite'],4), json.loads(l)['status']) for l in open('experiments/05_autoresearch_climate/results/experiment_log.jsonl')]"
```

### `best_config.json` — the current champion

For the reference loop this is **Exp5**: `composite: -1.7967…`, `test_primary: 1.7914…`,
`primary_metric: "rmse"`, `composite_fingerprint: "942df6a8450d8d57"`, and the full
champion `config` (`backbone: mlp`, `n_lags: 3`, `lr: 0.08`, `epochs: 1500`). It is
overwritten **only** when a later experiment beats the stored composite, so its
presence is the single source of truth for "what is currently best."

### `project_autoresearch_checkpoint.md` — crash recovery

A self-contained markdown the runner rewrites after **every** experiment. It records
the current champion (`Exp5 (mlp) composite=-1.7967`), the frozen fingerprint, the last
experiment's status/numbers (`Exp6 (mlp) — DISCARD`, per-fold test rmse `[1.8228]`,
elapsed `8.06s`), the **exact next command**, and a 3-step "How to resume". A power
failure costs at most one run because this file is current as of the last completed
experiment.

### `reasoning_annotations.json` — the audit trail

Keys are stringified experiment numbers; each value is a full `ReasoningEntry`
(pre-run diagnosis/citations/hypothesis/prediction + post-run verdict/learning). This
is what a referee reads to confirm every experiment carried a real citation, a
mechanism, and a numeric prediction *before* it ran.

---

## 5. Resuming a fresh session after a crash

Suppose the laptop lost power mid-loop. A brand-new Claude Code session (with no memory
of the previous one) recovers like this:

1. **Read the checkpoint.** `project_autoresearch_checkpoint.md` tells you the champion
   (`Exp5`, composite `-1.7967`), the frozen fingerprint (`942df6a8450d8d57`), the last
   completed experiment, and the exact next command.
2. **Tail the log.** `experiment_log.jsonl` confirms which experiment numbers actually
   completed (the runner derives the next number with `_next_experiment_num`, which is
   `max(experiment_num) + 1`, so a half-written run does not corrupt numbering).
3. **Re-read the last reasoning blob.** `reasoning_annotations.json` shows the last
   `learning` — i.e. the "next try" you had already committed to.
4. **Author the next pre-run entry** (step 2 of the checkpoint's resume list). It must
   pass both gates exactly as in §2.
5. **Run one experiment**, analyze, re-checkpoint. You are back in the loop.

Because the fingerprint is re-asserted on every run, a resumed session physically
cannot continue with a different metric definition — `assert_fingerprint` raises if the
composite was changed since Exp1. That is the Goodhart guard surviving a crash.

A quick way to *simulate* recovery for a demo: delete the in-memory state by starting a
new shell, then run only the checkpoint's "Exact next command" path. The artifacts on
disk are the entire state.

---

## 6. The real Exp05 champion progression (what "success" looks like)

Six experiments, one attributable change each, on synthetic daily 2m-temperature
(14 years, seed 0), primary metric **RMSE** (frozen), composite
`min(val, test) − 0.5·n_below(−12.0)`, fingerprint `942df6a8450d8d57`. Walk-forward
folds (`n_blocks=7, n_initial=3, gap=1`) for Exp1–5; a super-fold-by-regime split for
the Exp6 robustness check.

| Exp | one attributable change | composite (−RMSE) | test RMSE (°C) | skill vs persistence | status |
|----:|--------------------------|------------------:|---------------:|---------------------:|--------|
| 1 | persistence floor (walk-forward by year) | −1.9586 | 1.9477 | +0.000 | KEEP |
| 2 | tiny MLP, 7 lags + sin/cos day-of-year | −1.8503 | 1.8435 | +0.054 | KEEP |
| 3 | shorter **3-lag** window (parsimony) | −1.8253 | 1.8237 | +0.064 | KEEP |
| 4 | learning-rate tune 0.05 → 0.08 | −1.8157 | 1.8129 | +0.069 | KEEP |
| 5 | more epochs 400 → 1500 (full convergence) | **−1.7967** | **1.7914** | **+0.080** | KEEP |
| 6 | super-fold-by-regime robustness | −1.9622 | 1.8228 | +0.064 | DISCARD |

**Final champion: Exp5.** The champion composite moves monotonically
**−1.9586 → −1.8503 → −1.8253 → −1.8157 → −1.7967**, then Exp6 correctly DISCARDs
(a harder split scores worse, so the champion is left in place). That is a
**+8.0% RMSE improvement over persistence** (skill `+0.080`).

**Read it honestly.** One-step daily-temperature forecasting is a near-persistence
problem, so the learned model beats the naive floor only *modestly*, and most of the
gain is the seasonal encoding plus full convergence — not architecture. The loop says
so. It also caught a wrong "obvious" idea: **more** lags would have *hurt* this
near-AR(1) series, so Exp3 went the other way (3 lags) on a parsimony argument
(Box-Jenkins 2015, Wiley) and won. A hurried manual sweep would likely have shipped the
7-lag version. The value is not a heroic number; it is that every modest delta arrived
gated, cited, mechanistic, and Goodhart-proof.

---

## 7. Scaling up on the RTX 4090

The CPU smoke path is the runs-anywhere headline: a tiny numpy MLP, synthetic ERA5
stand-in, no keys, whole loop in ~38 s. On the laptop's **RTX 4090 (~16 GB)** the
**protocol is identical** — gate every experiment, freeze the composite, judge on
future years and the worst regime. Only two things change:

- **Backbones scale up.** The backbone dispatch in `autoresearch_env/runner.py`
  (`if backbone == "persistence" … elif backbone == "mlp" …`) is the **single extension
  point**. Add an `elif backbone == "lstm":` branch for a torch LSTM
  (Hochreiter & Schmidhuber 1997, *Neural Computation* — the longer-memory model the
  Exp6 learning explicitly flagged as the next try), or a zero-shot TS foundation model:
  **TimesFM** (Das et al. 2024 ICML, arXiv:2310.10688) or **Chronos**
  (Ansari et al. 2024 TMLR, arXiv:2403.07815). Prefer **using** these pretrained models
  zero-shot over writing a new architecture. The gates, composite, fingerprint, splits,
  logging, and checkpoint are untouched.
- **Data scales up.** Point `load_dataset` at real **ERA5** (see
  `common.synthetic_climate.REAL_DATA_NOTE` for the cdsapi/xarray swap) or many
  station-years. Walk-forward-by-year and super-fold-by-regime carry over directly to a
  real, non-stationary, warming series.
- **Budget & resilience.** Set a per-experiment timeout; the checkpoint is rewritten
  after every run, so a laptop crash still costs at most one experiment.

The headline does not change with hardware: an agent fast enough to run the loop for
days (or unattended overnight), held to a bar a referee would recognise.

---

## 8. Don't skip the human sign-off

The loop accelerates the *search*; it does not transfer scientific accountability. Before
any champion number reaches a slide or a paper, run the **`env-stats-validate`** skill
(`skills/env-stats-validate/SKILL.md`) against it: leakage (temporal **and** spatial)
actively ruled out, an honest baseline + skill score reported, uncertainty quantified
and its coverage checked, multiple-testing corrected (an autoresearch loop that ran many
experiments has done implicit multiple comparisons — confirm the winner on a held-out
split it was never selected on), reproducibility artifacts committed, every AI-supplied
citation verified or `(verify)`-flagged, and a **named human statistician signs off**.
AI proposes; the human decides.

### Pointers
- Protocol + worked example: [`docs/autoresearch_protocol.md`](autoresearch_protocol.md)
- The ladder (L1→L4): [`docs/zero_to_hero.md`](zero_to_hero.md)
- Runnable package: [`autoresearch_env/README.md`](../autoresearch_env/README.md)
- Gates: [`autoresearch_env/reasoning.py`](../autoresearch_env/reasoning.py)
- Flagship: [`experiments/05_autoresearch_climate/README.md`](../experiments/05_autoresearch_climate/README.md)
- Rigor checklist: [`skills/env-stats-validate/SKILL.md`](../skills/env-stats-validate/SKILL.md)
- Source methodology: `dlmastery/autoresearch` → `generalized_ml_autoresearch`
