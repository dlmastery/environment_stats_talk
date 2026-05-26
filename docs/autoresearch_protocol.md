# The Env-Stats AutoResearch Protocol

**Audience:** environmental statisticians (TIES-type) — skeptical of hype, value
statistical validity and reproducibility. **Purpose:** explain how an autonomous,
literature-grounded research loop is adapted from a general ML framework to
environmental statistics, and how it runs locally on a single GPU.

**Source credit.** This protocol is adapted from the user's own
`dlmastery/autoresearch` repository, specifically its domain-agnostic
`generalized_ml_autoresearch` package (a successor to an FX-specific AutoResearch
loop). The 7-step loop, the citation/reasoning gates, the
`min(val,test) − penalty·n_below_threshold` composite, the pluggable
backbones/splits, the winner archive, and the crash-recovery checkpoint are all
borrowed from there. Our contribution is the **environmental-statistics adaptation**:
env composite metrics, env splits, and a citation gate seeded with the real
env-stats SOTA catalog. The runnable, self-contained version lives in this repo at
[`autoresearch_env/`](../autoresearch_env/README.md).

**Anchor discipline (scientists' audience).** Only verifiable papers/tools are
cited. arXiv IDs are given for the five named env-AI papers; TS foundation models
are named without invented version numbers or dates. Anything not independently
confirmed is marked **(verify)** against `ledgers/CITATIONS-TO-VERIFY.md` before it
reaches a slide.

---

## 1. Why an env-statistician should care

Traditional environmetrics is a careful, *manual* loop: diagnose a series, reach for
a method (ARIMA, GEV, kriging), fit it, check residuals, iterate. The bottleneck is
not compute — it is the researcher's time spent wiring pipelines and reading
literature. The autoresearch loop keeps the researcher's *judgement* (model choice,
assumptions, leakage, multiple-testing, uncertainty) but hands the *wiring* to an
agent that must justify every move against the literature. The "100× faster" claim
is about removing the wiring tax, **not** about removing rigor — the gates below
make the loop *more* disciplined than a typical manual sweep, not less.

The same idea is now appearing in the literature: agentic systems that propose and
test hypotheses — "Towards an AI Co-Scientist" (arXiv:2502.18864) — and end-to-end
automated discovery — "The AI Scientist-v2" (arXiv:2504.08066). Our loop is a
*scoped, auditable* instance of that idea for environmental forecasting, with
Claude Code as the reasoning engine.

## 2. The 7-step loop, applied to environmental statistics

**Claude Code is the outer loop** — there is no separate Python "agent". Each
iteration runs the same seven steps the source framework enforces:

| # | Step | What it means for env-stats |
|---|------|-----------------------------|
| 1 | **Diagnose** | Read the experiment log + checkpoint; find the champion's weakest *fold/year* or *climate regime* (e.g. "skill collapses in the warm-phase regime"). |
| 2 | **Cite** | Pick a *real* reference that addresses the weakness, from `autoresearch_env/sota_catalog_env.yaml` or the verified anchors in `CLAUDE.md` §6. The citation must carry author/year/venue/title-or-arXiv + a relevance note. |
| 3 | **Hypothesize** | State the change *mechanistically* — "...because the mechanism is longer-memory persistence than a 7-lag window can encode, per Hochreiter & Schmidhuber 1997." |
| 4 | **Predict** | Commit a *numeric range* for the expected change — "skill-vs-persistence +0.05 to +0.12 on the cool-phase regime." |
| 5 | **Execute** | Run **exactly one** experiment via the runner. One change per experiment, so the diff is attributable. |
| 6 | **Analyze** | Compare the realized composite to the prediction; write a verdict (KEEP / DISCARD / NEAR-MISS) and a learning (axis open/closed, next try). |
| 7 | **Checkpoint** | The runner rewrites a self-contained crash-recovery markdown after every experiment so a power loss costs at most one run. |

Steps 2–4 are not optional prose — they are **machine-checked** before step 5 can run.

## 3. The two hard gates (why this is rigorous, not vibes)

Both gates are implemented in
[`autoresearch_env/reasoning.py`](../autoresearch_env/reasoning.py) and wired into
the runner, which **refuses to launch** an experiment that fails either one.

### 3a. Citation-Rigor gate
A citation passes only if it contains: a 4-digit **year**; a recognized **venue**
(env-stats + ML vocabulary — arXiv, ICML/ICLR/NeurIPS, JGR, GMD, ERL, WAF, Climate
Dynamics, Technometrics, Environmetrics, plus established publishers OTexts/Springer/
Wiley for the classical-baseline textbooks); an **arXiv ID or a quoted title**; and a
one-clause **relevance note**. Bare tags like `(ClimateLLM2025)` are rejected. A
single-paper citation must clear a 40-word floor (80 words for multi-paper). This is
exactly the source framework's gate, with the venue vocabulary re-pointed at the
env-stats literature.

### 3b. Reasoning-Completeness gate
Each reasoning field has a word-count floor (diagnosis ≥ 60, hypothesis ≥ 50,
prediction ≥ 25, verdict ≥ 30, learning ≥ 40) and a required-keyword set. The
**hypothesis must be mechanistic** (contains "mechanism" / "because" / "per …"); the
**prediction must contain a numeric range** (single-point predictions are rejected);
the **verdict must say KEEP/DISCARD/NEAR-MISS**; the **learning must mark an axis
open/closed** or a "next try". These floors are the mechanism that prevents drift
into "let me just try X" guessing.

## 4. Env composite metrics

The composite is the *single number* the loop optimizes; a new experiment is kept
only if its composite beats the champion's. The default, inherited from the source
framework, is:

```
composite = min(val_primary, test_primary) − penalty_weight · n_below_threshold
```

- `min(val, test)` rewards the **worse** of validation/test, penalising a model that
  overfits to either split.
- `n_below_threshold` counts **folds/regimes that fail an acceptance bar**, so a
  model strong on average but collapsing on one regime is penalised — the property
  env-stats reviewers care about most.

The **primary metric** is environmental, not generic:

| Primary metric | What it is | Orientation |
|---|---|---|
| **latitude-weighted RMSE** | gridded-field RMSE with each cell weighted by `cos(lat)` (poles cover less area), the ERA5-benchmark convention; `common.metrics.latitude_weighted_rmse`. On a single-station series it reduces to plain RMSE. | lower is better |
| **ACC** (anomaly correlation) | correlation of forecast vs observed anomalies about a climatology; the standard NWP skill score; operational target ACC > 0.6; `common.metrics.anomaly_correlation`. | higher is better |
| **skill-vs-persistence** | `1 − RMSE_model / RMSE_persistence`; 0 = no better than "tomorrow = today", >0 = real skill; `common.metrics.skill_score`. | higher is better |

The calculator orients error metrics (negates internally) so a larger composite is
always "better" regardless of metric direction. See
[`autoresearch_env/composite.py`](../autoresearch_env/composite.py).

### Goodhart protection: the frozen fingerprint
A classic failure mode is silently rewriting the metric mid-project to "improve" it.
`CompositeCalculator.fingerprint()` hashes the metric name + orientation + penalty +
threshold; the runner stores it on the first experiment and calls
`assert_fingerprint(frozen)` on every subsequent one. If anyone changes the metric
definition, the fingerprint diverges and the runner **raises** rather than logging an
incomparable number. (Reward-hacking the test set — shrinking/shifting it — is
forbidden by the same discipline: the split protocol and indices are frozen.)

## 5. Env splits (leakage-resistant, regime-aware)

Implemented in [`autoresearch_env/splits.py`](../autoresearch_env/splits.py):

- **Walk-forward by year** (`WalkForwardSplit`). The series is cut into contiguous
  blocks (one per year). Fold *k* trains on all earlier blocks and tests on a future
  block never seen in training, with a `gap` purge at the seam to block lagged-target
  leakage. This is the honest evaluation for a **warming, non-stationary** climate —
  the model must *extrapolate forward in time*, exactly as an operational forecast
  must. Expanding or fixed-rolling windows are both supported.
- **Super-fold by climate regime** (`SuperFoldSplit`). Samples carry an integer
  regime label (here a synthetic ENSO-like warm/neutral/cool phase). The splitter
  holds out the most-recent slice of *each* regime at once and trains on everything
  else, producing **one** fold whose test set spans **all** regimes. Combined with
  the composite's per-fold penalty, this judges a model on its **worst regime**, not
  its average — the analogue of the FX framework's "super-fold over regime windows".
- **Spatial GroupKFold** (conceptual, for gridded data). When the unit is a spatial
  cell/station rather than a time step, group folds by spatial block so no cell
  appears in both train and test — the standard guard against spatial autocorrelation
  inflating skill. (The runnable package ships the two time-axis splitters above; the
  source `generalized_ml_autoresearch` provides `GroupKFold` for the gridded case.)

`validate_no_overlap` asserts train/val/test disjointness within every fold and
raises on any violation — the programmatic form of "never leak the future".

## 6. The citation gate's SOTA catalog (real anchors only)

The gate is *seeded* by [`autoresearch_env/sota_catalog_env.yaml`](../autoresearch_env/sota_catalog_env.yaml),
the env-stats analogue of the source repo's `templates/sota_catalog.yaml`. Every
entry carries a full, verifiable citation. The menu the loop draws from:

- **Classical baselines (the floors every model must beat):** persistence &
  seasonal-naive and ARIMA (Hyndman & Athanasopoulos 2021, OTexts; Box-Jenkins 2015,
  Wiley); Mann-Kendall + Sen's slope for trend (Mann 1945; Sen 1968); GEV / POT for
  extremes (Coles 2001, Springer).
- **Neural baselines:** the atmospheric-sciences MLP (Gardner & Dorling 1998,
  *Atmospheric Environment*) and the LSTM (Hochreiter & Schmidhuber 1997, *Neural
  Computation*).
- **TS foundation models (zero-shot AFTER story):** **TimesFM** (Das, Kong, Sen &
  Zhou 2024 ICML, arXiv:2310.10688), **Chronos** (Ansari et al. 2024 TMLR,
  arXiv:2403.07815), **MOMENT** (Goswami et al. 2024 ICML, arXiv:2402.03885).
- **Climate LLMs & agents:** **ClimateLLM** (arXiv:2502.11059), **CLLMate**
  (arXiv:2409.19058), **AI co-scientist** (arXiv:2502.18864), **AI-Scientist-v2**
  (arXiv:2504.08066), **EagleVision** (arXiv:2503.23330). These are the conceptual
  anchors for using an LLM as the outer loop and for the multimodal/event-aware and
  remote-sensing tracks. Exact venues/dates are flagged **(verify)** per `CLAUDE.md`
  §6 and the citations ledger.

## 7. How it runs on a 4090 (and why it runs anywhere first)

The package's default path is **CPU, synthetic, no keys**: `runner.load_dataset`
builds lag features from `common.daily_temperature` (deterministic synthetic ERA5
stand-in), and the default backbone is a tiny numpy MLP that trains in milliseconds.
This is the runs-anywhere headline-result discipline — anyone can reproduce it
offline.

On the laptop's **RTX 4090 (~16 GB)**, only the backbone and data scale up; the
*protocol is identical*:

- **Backbones scale up.** Swap the numpy MLP for a torch LSTM, or a **TimesFM /
  Chronos zero-shot** forecaster (catalog tiers 2–3). The backbone dispatch in
  `runner.py` is the single extension point; the gates, composite, splits, logging,
  and checkpoint are untouched.
- **Data scales up.** Point `load_dataset` at real **ERA5** (see
  `common.synthetic_climate.REAL_DATA_NOTE` for the cdsapi/xarray swap) or many
  station-years; walk-forward-by-year and super-fold-by-regime carry over directly.
- **Resilience.** Checkpoint after *every* experiment (the repo's power-failure rule)
  so a laptop crash costs at most one run. A per-experiment timeout caps the budget.

## 8. Worked example — autonomously improving the Exp01 climate forecaster

Exp01 (`experiments/01_climate_timeseries_forecast/`) is the BEFORE: persistence,
seasonal-naive, and SARIMA one-step-ahead temperature forecasts. **Exp05** is the
flagship AFTER — the loop driving itself to beat that BEFORE. A representative
three-iteration trajectory:

**Iteration 1 — establish the floor.**
- *Diagnose:* no champion yet; goal is a defensible walk-forward RMSE/skill floor.
- *Cite:* Hyndman & Athanasopoulos 2021 (OTexts) — persistence is the skill
  reference.
- *Hypothesize:* "persistence will set the reference because at one-step-ahead the
  AR(1) weather term dominates."
- *Predict:* "walk-forward test RMSE 1.5–3.0 °C; skill-vs-persistence = 0 by
  definition for the reference itself."
- *Execute:* `backbone: persistence`, `split: walk_forward (by year)`,
  `primary_metric: lat_weighted_rmse`. → champion (first experiment always is).
- *Analyze/Checkpoint:* KEEP; "axis open for a learned model; next try: a tiny MLP on
  7 lags + seasonal encoding."

**Iteration 2 — a learned baseline.**
- *Diagnose:* persistence ignores the annual cycle's shape and the warming trend.
- *Cite:* Gardner & Dorling 1998 (*Atmospheric Environment*) — atmospheric MLP on
  lag/seasonal features.
- *Hypothesize:* "a 1-hidden-layer MLP on 7 lags + sin/cos day-of-year will cut RMSE
  *because* the seasonal encoding plus short-lag autocorrelation captures most of the
  one-step signal."
- *Predict:* "composite improves by +0.2 to +0.8 °C-equivalent; skill-vs-persistence
  +0.02 to +0.15."
- *Execute:* `backbone: mlp`. → KEEP **iff** its composite beats Iteration 1; the
  runner writes a new `best_config.json` only on improvement.
- *Analyze:* compare realized skill to the predicted range; if it lands short on the
  warm-phase regime, that becomes the next diagnosis.

**Iteration 3 — attack the weakest regime.**
- *Diagnose:* the super-fold-by-regime run shows the warm-phase fold below threshold
  (the per-regime penalty docked the composite).
- *Cite:* Hochreiter & Schmidhuber 1997 (*Neural Computation*) — LSTM longer memory;
  or a TS **foundation model** (TimesFM / Chronos) for a zero-shot probe on the 4090.
- *Hypothesize:* "longer memory than a 7-lag window will recover warm-phase skill
  *because* the regime's persistence horizon exceeds one week."
- *Predict:* "warm-phase fold skill +0.05 to +0.12, lifting `n_below_threshold` from
  1 to 0 and the composite accordingly."
- *Execute / Analyze / Checkpoint:* KEEP only if the worst-regime number actually
  improves — the composite refuses to credit an average-only gain.

Throughout, **the metric fingerprint is frozen**, so none of these "improvements" can
come from quietly redefining skill; **every** experiment carried a real citation, a
mechanistic hypothesis, and a numeric prediction *before* it was allowed to run; and
**every** result is one attributable change against the logged champion. That is the
env-stats autoresearch protocol: an agent fast enough to run the loop for days, held
to a bar a referee would recognize.

---

### Pointers
- Runnable package + how-to: [`autoresearch_env/README.md`](../autoresearch_env/README.md)
- Gates: [`autoresearch_env/reasoning.py`](../autoresearch_env/reasoning.py)
- Composite + fingerprint: [`autoresearch_env/composite.py`](../autoresearch_env/composite.py)
- Splits: [`autoresearch_env/splits.py`](../autoresearch_env/splits.py)
- SOTA catalog: [`autoresearch_env/sota_catalog_env.yaml`](../autoresearch_env/sota_catalog_env.yaml)
- Tests: `python -m pytest autoresearch_env/tests -q`
- Source methodology: `dlmastery/autoresearch` → `generalized_ml_autoresearch`
