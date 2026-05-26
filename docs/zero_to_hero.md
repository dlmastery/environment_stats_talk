# Zero to Hero — the Claude-Code ladder for environmental statistics

> **Two ideas run through every level.**
> 1. **USE existing SOTA — don't build from scratch.** The fastest correct path is almost always a *pretrained* AI-for-science model used zero-shot or lightly adapted (TimesFM / Chronos for time series, a structured-extraction LLM for text, a pretrained vision backbone for imagery), and *established* statistical methods (Mann-Kendall, GEV/POT, kriging) wired together by the agent — not a novel architecture you write yourself.
> 2. **The art is telling the agent what to do.** Your scientific leverage is in the *prompt and the spec*: the question, the data, the baseline to beat, the validation design, the rigor gates. The agent is fast at typing and running; *you* are the one who knows what "correct" means here.

The ladder goes from **no code** to a **self-improving autonomous research loop**. Each rung is useful on its own — most day-to-day work lives at L2/L3. Every level lists: *what it is*, a *copy-paste prompt template*, *tools*, *realistic time*, a *concrete env-stats example tied to a repo experiment*, and the *rigor checkpoint* you must not skip.

Verifiable anchors only: ClimateLLM (arXiv:2502.11059), CLLMate (arXiv:2409.19058), AI co-scientist (arXiv:2502.18864), AI-Scientist-v2 (arXiv:2504.08066), EagleVision (arXiv:2503.23330); ERA5, GBIF, Sentinel-2, CAMELS, OpenAQ; TimesFM, Chronos. Anything version/date-specific is `(verify)`; any number is `(measured in ExpNN)`.

---

## L1 — Zero-code prompting

**What it is.** You don't write or run anything. You use Claude as a *thinking partner*: ask it to propose hypotheses, draft an analysis plan, suggest the right statistical test, sketch (not run) the code, or critique your design. Output is text — ideas, a plan, a checklist.

**When it's enough.** Scoping a study; choosing a test; sanity-checking assumptions; turning a vague question into a falsifiable hypothesis; getting a literature-shaped starting point before you touch data.

**Copy-paste prompt template.**
```
You are an environmental statistician. I have <DATA: e.g., 40 yrs of daily ERA5 2m-temperature
at one gridpoint>. My question is <QUESTION: e.g., is there a significant warming trend, and
are extremes getting more frequent?>.
1. State 2–3 falsifiable hypotheses.
2. Recommend the appropriate statistical methods and name the assumptions each requires.
3. List the validation pitfalls I must avoid (autocorrelation, multiple testing, leakage).
4. Sketch (do NOT run) the analysis steps as a numbered plan.
Cite only well-established methods; if you are unsure of a fact, say "(verify)".
```

**Tools.** Claude chat / Claude Code in plan mode. No execution.

**Realistic time.** Minutes.

**Concrete example → `Exp02`.** Before writing any extremes/trend code, ask Claude to lay out Mann-Kendall + GEV/POT, name the autocorrelation and multiple-site pitfalls, and produce the analysis plan that `Exp02` then implements.

**Rigor checkpoint.** Treat output as *hypotheses and a plan, not findings*. Nothing here is evidence; every suggested method still needs its assumptions checked against your real data. Mark any factual claim `(verify)`.

---

## L2 — Tool-augmented Claude Code (writes **and runs** Python)

**What it is.** Claude Code now *executes*: it writes Python, fetches data, fits/runs models, validates, and shows you metrics and plots — iterating on errors itself. This is the workhorse rung. Crucially, you direct it to **use existing SOTA** (e.g., a TimesFM/Chronos zero-shot forecast as the AFTER baseline) rather than hand-rolling a model.

**Copy-paste prompt template.**
```
Use Claude Code with Python execution. Goal: <e.g., forecast monthly mean temperature 12 steps
ahead and beat a persistence/climatology baseline>.
Data: <synthetic generator in common/ OR small public sample, e.g., ERA5-style series>. No API
keys / no GPU required for the headline run; note any real-data or 4090 variant separately.
Steps:
1. Load the data; print shape, range, missingness.
2. Build an HONEST baseline first (persistence + climatology).
3. AFTER model: use an EXISTING foundation model zero-shot (TimesFM or Chronos (verify)); do NOT
   write a new architecture.
4. Validate with a TIME-SERIES split (walk-forward) — no leakage. Report RMSE/ACC with the
   baseline alongside, plus an uncertainty band.
5. Save metrics table + forecast/residual plots to results/. Use a fixed seed.
Stop and show me the validation design before reporting final numbers.
```

**Tools.** Claude Code + Python (`numpy`/`pandas`/`statsmodels`/`scikit-learn`), `common/` synthetic generators, foundation-model inference (CPU-ok zero-shot, or 4090 for the small NN variant), matplotlib (Agg).

**Realistic time.** Minutes to ~an hour, mostly compute and your review.

**Concrete examples → `Exp01` / `Exp02`.**
- `Exp01` (climate forecasting): baseline ARIMA/persistence vs **TimesFM/Chronos zero-shot** and a small LSTM/TCN — metrics + plots into `results/`.
- `Exp02` (extremes/trends): Mann-Kendall + GEV/POT pipeline with the rigor checklist run inline.

**Rigor checkpoint.** *You* approve the validation design **before** trusting numbers: confirm the split (walk-forward, no leakage), an honest baseline is present, uncertainty is reported, and the seed is fixed. Re-run to confirm reproducibility.

---

## L3 — Multi-agent (a team of specialized agents)

**What it is.** Instead of one agent doing everything serially, you orchestrate a small **hierarchy of SME agents**, each with one job, integrated by a lead:
- **Data agent** — fetches/cleans data (ERA5 / GBIF / Sentinel-2 / OpenAQ / CAMELS), documents provenance.
- **Modeling agent** — fits the baseline + the SOTA AFTER model.
- **Validation agent** — runs the rigor checklist (splits, multiple-testing, uncertainty, leakage) as an *independent* check on the modeler.
- **Reporting agent** — writes the results README / brief from the committed artifacts.

The win is *parallelism + separation of concerns*: the validator is not the modeler, which catches over-optimistic results.

**Pattern / prompt template.**
```
Act as a lead orchestrator over four SME sub-agents. Task: <e.g., extract species interactions
from text and build the interaction network; OR classify land cover from Sentinel-2 tiles>.
- Data agent: acquire + clean <GBIF text / Sentinel-2 tiles>; emit a provenance note + a held-out
  set the modeler never sees.
- Modeling agent: build an HONEST baseline (regex/keyword | RF on spectral indices) AND the AFTER
  model (structured-LLM extraction | CNN on 4090) — reuse EXISTING models, don't invent.
- Validation agent (independent): score precision/recall vs gold | per-class F1 with spatially-
  blocked CV; flag leakage, imbalance, multiple-testing.
- Reporting agent: write results/README.md from committed metrics/plots only — no new claims.
Integrate, resolve disagreements between modeler and validator, and show me the validation design
before final numbers.
```

**Tools.** Claude Code multi-agent / sub-agent orchestration; the `env-stats-validate`, `climate-data-fetch`, and report-writer skills; 4090 for the heavy modeling agent.

**Realistic time.** Tens of minutes to a couple of hours; agents run in parallel.

**Concrete examples → `Exp03` / `Exp04`.**
- `Exp03` (biodiversity-from-text): data→extraction→validation→report, structured-LLM extraction in the spirit of **CLLMate (arXiv:2409.19058)**; scored precision/recall vs a gold set.
- `Exp04` (remote sensing): RF-on-indices baseline vs **CNN on the 4090**, detection/scene framing related to **EagleVision (arXiv:2503.23330)** `(verify scope)`; spatially-blocked CV by the independent validator.

**Rigor checkpoint.** The **validation agent must be independent of the modeling agent**, and its checklist (gold-set scoring / spatial-block CV, class imbalance, multiple-testing, uncertainty) gates the report. The reporting agent may only restate committed artifacts — no fresh claims.

---

## L4 — Self-improving autoresearch loop (on a 4090)

**What it is.** The flagship. **Claude Code is the researcher**, running a closed **7-step loop — Diagnose → Cite → Hypothesize → Predict → Execute (one experiment) → Analyze → Checkpoint** — over **N experiments autonomously**, with **champion tracking** (a self-contained best-model archive), a **research journal/audit trail**, and **hard gates**: a citation-rigor gate and a reasoning-completeness gate must pass *before* any experiment runs, and a composite-metric fingerprint catches mid-project goal drift. Methodology follows **AI co-scientist (arXiv:2502.18864)** and **AI-Scientist-v2 (arXiv:2504.08066)**; the implementation adapts the user's `dlmastery/autoresearch` `generalized_ml_autoresearch` loop to env-stats (env composite metrics like latitude-weighted RMSE/ACC, env splits like spatial GroupKFold and walk-forward-by-year).

**Copy-paste prompt template.**
```
Run the env-stats autoresearch loop (autoresearch_env/) on the 4090. Objective: <e.g., maximize a
latitude-weighted skill metric on the Exp01 forecasting task without leakage>.
Constraints:
- One experiment per iteration; follow Diagnose→Cite→Hypothesize→Predict→Execute→Analyze→Checkpoint.
- Hard gates: citation-rigor + reasoning-completeness MUST pass before each run; abort the run if not.
- Composite metric = min(val, test) − penalty·n_below_threshold; fingerprint it to detect goal drift.
- Use the env split (walk-forward by year / spatial GroupKFold); reuse EXISTING backbones (TimesFM/
  Chronos/LSTM stubs) — do not invent architectures.
- Track a champion (self-contained archive + audit + inference); checkpoint after every iteration for
  crash recovery. Budget: N iterations. Report the journal and the champion at the end.
```

**Tools.** `autoresearch_env/` loop, Claude Code as the driver, RTX 4090 (Laptop, ~16 GB), foundation-model backbones, crash-recovery checkpointing, the autoresearch-setup skill.

**Realistic time.** Hours, **unattended** — the biggest gain here is throughput while you're away.

**Concrete example → `Exp05`.** The loop runs N forecasting/extremes experiments on the 4090, tracks the champion, and emits the research journal + champion archive committed to `results/`. This is the literal "science 100× faster" demo `(speedup measured in Exp05)`.

**Rigor checkpoint.** The **gates run before compute, not after**: no citation/reasoning → no experiment. The champion is only valid on a **held-out split**, the composite-metric fingerprint must be unchanged (no silent goal rewrite), every iteration is checkpointed, and **a human signs off the final claims before they reach a slide or a paper.** Autonomy speeds the search; it does not transfer scientific accountability.

---

## The ladder at a glance

| Level | What runs | You provide | Realistic time | Repo anchor |
|---|---|---|---|---|
| **L1** Zero-code prompting | Nothing (text only) | The question + judgment | Minutes | plan for `Exp02` |
| **L2** Tool-augmented Claude Code | Agent writes & runs Python | The spec + validation design | Minutes–1h | `Exp01`, `Exp02` |
| **L3** Multi-agent team | Parallel SME agents (data/model/validate/report) | The orchestration + rigor gates | Tens of min–hours | `Exp03`, `Exp04` |
| **L4** Autoresearch loop | Claude Code runs N experiments autonomously | The objective + hard gates + sign-off | Hours, unattended | `Exp05` (flagship) |

**The through-line:** climb only as high as the task needs, always **use existing SOTA**, and at every rung **the human owns the science** — the art is in *telling the agent what to do*, and in the rigor checkpoints you refuse to skip.
