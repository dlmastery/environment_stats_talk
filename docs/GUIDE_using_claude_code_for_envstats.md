# Guide — Using Claude Code for Environmental Statistics

### The art of telling the agent what to do

**Audience:** a working environmental statistician (TIES-type) — busy, skeptical of
hype, with limited patience for tooling. This guide is pragmatic. It does not teach
you statistics; it teaches you how to *direct an agent* so that the statistics you
already know come out correct and fast.

**The two ideas that run through everything:**

1. **USE existing SOTA — don't build from scratch.** The fastest correct path is
   almost always a *pretrained* AI-for-science model used zero-shot or lightly adapted
   (TimesFM / Chronos for time series, a structured-extraction LLM for text, a
   pretrained vision backbone for imagery) plus *established* methods (Mann-Kendall,
   GEV/POT, kriging) wired together by the agent — not a novel architecture you ask it
   to invent.
2. **Your leverage is the prompt and the spec.** The agent is fast at typing and
   running; *you* are the one who knows what "correct" means here — the question, the
   data, the baseline to beat, the validation design, the rigor gates, and the
   sign-off. The agent proposes; you decide.

This guide pairs with the **L1→L4 ladder** in [`docs/zero_to_hero.md`](zero_to_hero.md)
and the **rigor checklist** in
[`skills/env-stats-validate/SKILL.md`](../skills/env-stats-validate/SKILL.md). Climb only
as high as the task needs; most day-to-day work lives at L2/L3.

---

## 1. The ladder in practice (with copy-paste prompts)

| Level | What runs | You provide | Time | Repo anchor |
|---|---|---|---|---|
| **L1** Zero-code prompting | Nothing (text only) | The question + judgment | Minutes | plan for `Exp02` |
| **L2** Tool-augmented Claude Code | Agent writes & runs Python | The spec + validation design | Minutes–1h | `Exp01`, `Exp02` |
| **L3** Multi-agent team | Parallel SME agents (data/model/validate/report) | The orchestration + rigor gates | Tens of min–hours | `Exp03`, `Exp04` |
| **L4** Autoresearch loop | Claude Code runs N experiments autonomously | The objective + hard gates + sign-off | Hours, unattended | `Exp05` (flagship) |

### L1 — Zero-code prompting (thinking partner)

Use it to scope a study, pick a test, sanity-check assumptions, turn a vague question
into a falsifiable hypothesis. Output is **a plan and hypotheses, not findings.**

```
You are an environmental statistician. I have <DATA: e.g. 40 yrs of daily ERA5 2m-temperature
at one gridpoint>. My question is <QUESTION: e.g. is there a significant warming trend, and are
extremes getting more frequent?>.
1. State 2–3 falsifiable hypotheses.
2. Recommend the appropriate methods and name the assumptions each requires.
3. List the validation pitfalls I must avoid (autocorrelation, multiple testing, leakage).
4. Sketch (do NOT run) the analysis steps as a numbered plan.
Cite only well-established methods; if you are unsure of a fact, say "(verify)".
```

*Rigor checkpoint:* nothing here is evidence. Every suggested method still needs its
assumptions checked against your real data; mark any factual claim `(verify)`.

### L2 — Tool-augmented Claude Code (writes AND runs Python) — the workhorse

```
Use Claude Code with Python execution. Goal: <e.g. forecast monthly mean temperature 12 steps
ahead and beat a persistence/climatology baseline>.
Data: <synthetic generator in common/ OR a small public sample>. No API keys / no GPU for the
headline run; note any real-data or 4090 variant separately.
Steps:
1. Load the data; print shape, range, missingness.
2. Build an HONEST baseline FIRST (persistence + climatology).
3. AFTER model: use an EXISTING foundation model zero-shot (TimesFM or Chronos (verify)); do NOT
   write a new architecture.
4. Validate with a TIME-SERIES split (walk-forward) — no leakage. Report RMSE/ACC WITH the
   baseline alongside, plus an uncertainty band.
5. Save the metrics table + forecast/residual plots to results/. Use a fixed seed.
STOP and show me the validation design before reporting final numbers.
```

The last line is load-bearing: it forces a human checkpoint on the split *before* you
are tempted to believe a number.

### L3 — Multi-agent team (separation of concerns)

The win is parallelism + an **independent validator** (not the modeler), which catches
over-optimistic results.

```
Act as a lead orchestrator over four SME sub-agents. Task: <e.g. classify land cover from
Sentinel-2 tiles>.
- Data agent: acquire + clean the tiles; emit a provenance note + a held-out set the modeler
  never sees.
- Modeling agent: build an HONEST baseline (RF on spectral indices) AND the AFTER model (a
  pretrained CNN on the 4090) — REUSE existing models, don't invent.
- Validation agent (INDEPENDENT of the modeler): per-class F1 with spatially-blocked CV; flag
  leakage, class imbalance, multiple testing.
- Reporting agent: write results/README.md from committed metrics/plots ONLY — no new claims.
Integrate, resolve modeler-vs-validator disagreements, and show me the validation design before
final numbers.
```

*Rigor checkpoint:* the validation agent must be independent of the modeling agent; the
reporting agent may only restate committed artifacts.

### L4 — Self-improving autoresearch loop (the flagship)

Claude Code is the researcher, running the gated 7-step loop over N experiments. See
[`docs/TUTORIAL_autoresearch.md`](TUTORIAL_autoresearch.md) for the hands-on version.

```
Run the env-stats autoresearch loop (autoresearch_env/) on the 4090. Objective: <e.g. maximize a
latitude-weighted skill metric on the Exp01 forecasting task without leakage>.
Constraints:
- One experiment per iteration; Diagnose→Cite→Hypothesize→Predict→Execute→Analyze→Checkpoint.
- Hard gates: citation-rigor + reasoning-completeness MUST pass before each run; abort if not.
- Composite = min(val,test) − penalty·n_below_threshold; fingerprint it to detect goal drift.
- Env split (walk-forward by year / super-fold by regime); REUSE existing backbones (TimesFM/
  Chronos/LSTM) — do not invent architectures.
- Track a champion + checkpoint after every iteration. Budget: N iterations. Report the journal
  and the champion at the end.
```

*Rigor checkpoint:* the gates run **before** compute, not after; the champion is valid
only on a held-out split; the fingerprint must be unchanged; **a human signs off the
final claims before they reach a slide or a paper.** Autonomy speeds the search; it does
not transfer accountability. (The reference loop's honest result: a tiny MLP beats
persistence by ~8% RMSE — `Exp05`, measured.)

---

## 2. Principles for directing the agent

These apply at every rung. They are the difference between a fast-and-wrong demo and a
fast-and-defensible result.

1. **Decompose the task.** Give the agent one job at a time: load+describe, then
   baseline, then SOTA model, then validate, then report. A monolithic "build me a
   land-cover classifier" prompt invites it to skip the baseline and the split. Number
   your steps; the agent will follow the structure you impose.
2. **Give it the triple: data + question + constraints.** "Forecast temperature" is
   under-specified. "Forecast daily 2m-temperature one step ahead on this 14-year
   synthetic series, beat persistence, walk-forward split with a 1-step gap, fixed seed,
   report RMSE and skill-vs-persistence with an uncertainty band" is a spec the agent
   can execute correctly.
3. **Demand a baseline AND a skill score — always.** A "low RMSE" alone is meaningless.
   Require persistence / climatology (forecasting) or majority-class / RF-on-indices
   (classification), and a skill score (`common/metrics.skill_score`): 1 = perfect,
   0 = no better than the reference, < 0 = worse. If the agent reports a metric without
   the reference, send it back.
4. **Ask for uncertainty.** A point prediction is not a scientific result. Require CIs,
   prediction/conformal intervals, or ensemble spread, and a **coverage check**
   (empirical vs nominal). For EVT, require return-level CIs (bootstrap or
   profile-likelihood) and the shape parameter *with* its uncertainty.
5. **Make the agent show the validation design before the numbers.** Put
   "STOP and show me the split before reporting final metrics" in the prompt. You are
   the leakage referee; you cannot referee a number you have already anchored on.
6. **Pin reproducibility up front.** Fixed seeds, recorded package versions, data
   provenance, runs-anywhere on synthetic data with no key. Ask for it in the prompt so
   it is built in, not bolted on.
7. **Keep human sign-off.** Nothing reaches a slide until `env-stats-validate` passes and
   a named statistician approves. The agent proposes; you decide.

---

## 3. USE existing SOTA — don't build from scratch

The single highest-leverage instruction you can give. When the agent reaches for a model:

- **Time series:** zero-shot **TimesFM** (Das et al. 2024 ICML, arXiv:2310.10688) or
  **Chronos** (Ansari et al. 2024 TMLR, arXiv:2403.07815) as the AFTER baseline, against
  persistence/climatology/ARIMA as the BEFORE. Do not ask it to write a new forecasting
  architecture.
- **Trends/extremes:** established methods — Mann-Kendall + Sen's slope; GEV/POT
  (Coles 2001, Springer). The agent's job is to *wire and check* them, not to invent an
  estimator.
- **Text → structured data:** a structured-extraction LLM in the spirit of CLLMate
  (arXiv:2409.19058), scored against a gold set — not a bespoke parser.
- **Imagery:** a pretrained vision backbone / detection model (e.g. the EagleVision
  framing, arXiv:2503.23330 `(verify scope)`) on the 4090, against an RF-on-spectral-
  indices baseline.

Phrase it explicitly: *"Reuse an existing pretrained model zero-shot; do NOT write a new
architecture."* It keeps the agent on the correct, fast path and keeps your result
comparable to literature the audience trusts. Anything version/date-specific stays
`(verify)` until confirmed against `ledgers/CITATIONS-TO-VERIFY.md`.

---

## 4. Common failure modes and how to prevent them

These are the ways an agentic env-stats result goes wrong. Each has a prompt-level and a
check-level defence.

### Hallucinated citations
*Symptom:* a confident reference with a plausible title, venue, and date that does not
exist — or a real paper mis-dated/mis-attributed.
*Prevent:* tell the agent up front "cite only verifiable papers; mark anything unconfirmed
`(verify)` and route it to `ledgers/CITATIONS-TO-VERIFY.md`; never assert a date/venue you
cannot confirm." In the L4 loop the **Citation-Rigor gate** enforces a year + venue +
arXiv/title + relevance clause and rejects bare tags. Outside the loop, *you* verify every
citation before it reaches a slide. (Per `CLAUDE.md` §6, claims like "Nature 19 May 2026"
or "GPT-5.4" are flagged and never asserted.)

### Data leakage (the most common way an impressive result is wrong)
*Symptom:* suspiciously high skill; a model that beats a foundation-model zero-shot by an
implausible margin; near-perfect tail fits.
*Prevent:* mandate a **walk-forward** split for forecasting (no random k-fold on a time
series), a **purge + embargo** sized to the label horizon at split seams, and **spatial
GroupKFold** for gridded/station data (hold out whole basins/cells/regions). Insist that
scalers/imputers/climatology baselines are fit on **training only, inside the fold**. The
repo's splitters (`autoresearch_env/splits.py`) call `validate_no_overlap` and raise on any
train/val/test overlap. When you see implausible skill, *suspect leakage first.*

### Goodhart's law (optimising the scorer instead of the science)
*Symptom:* the metric quietly changes mid-project so the number "improves."
*Prevent:* freeze the metric. The L4 composite has a **frozen fingerprint**
(`CompositeCalculator.fingerprint()` — `942df6a8450d8d57` in the reference loop) hashed on
the first experiment and re-asserted on every later one; a silent redefinition *raises*
rather than logging an incomparable number. Outside the loop, write the metric and the
acceptance bar down before you run anything, and refuse to renegotiate them to fit a result.

### p-hacking and implicit multiple testing
*Symptom:* per-cell p < 0.05 across a large grid; a "best" model selected from many runs and
then reported as if it were pre-registered.
*Prevent:* require a **multiple-testing correction** (Bonferroni or FDR/Benjamini-Hochberg)
and state the family of tests and the corrected threshold. Recognise that an autoresearch
loop running many experiments has done **massive implicit multiple comparisons** — the
reported "best" is selection-biased, so **confirm the winner on a held-out fold it was never
selected on**. The L4 composite's `min(val, test)` and per-fold penalty push against
selecting a model that is good on average but collapses on one regime.

### Reporting the average and burying the bad regime
*Symptom:* a strong mean skill that hides a regime where the model fails.
*Prevent:* judge on the **worst regime**, not the average (super-fold-by-regime + per-fold
penalty), and report modest deltas as modest. The reference loop does exactly this: Exp6's
super-fold stress test scored worse than the champion and was correctly **DISCARDed** rather
than rationalised.

---

## 5. The `env-stats-validate` skill — your sign-off gate

Before any AI-produced result is reported, run
[`skills/env-stats-validate/SKILL.md`](../skills/env-stats-validate/SKILL.md) (trigger it
with `/env-stats-validate` or "review this AI-produced result for rigor"). It is a
human-in-the-loop **checklist, not an automation** — for each item you mark
**PASS / FAIL / N-A** with a one-line note and a pointer to the evidence. The eight checks:

1. **Data leakage** — temporal *and* spatial split correctness; preprocessing fit on
   training only.
2. **Stationarity & autocorrelation** — trend/variance checks; ACF/PACF or Ljung-Box on
   residuals; block bootstrap / HAC SEs where needed.
3. **Distributional assumptions** — env data is rarely Gaussian; EVT threshold/block-size
   justification; report the shape parameter with uncertainty.
4. **Multiple-testing correction** — Bonferroni / FDR; beware implicit multiplicity from
   many experiments.
5. **Uncertainty quantification** — CIs / conformal intervals + a coverage check.
6. **Baseline / skill-score comparison** — an honest baseline and a reported skill score,
   field-appropriate metric (latitude-weighted RMSE / ACC).
7. **Reproducibility** — fixed seeds, pinned versions, data provenance, runs-anywhere.
8. **AI-specific** — no invented facts; Goodhart/fingerprint integrity; prompt-induced bias
   re-checked against the raw evidence, not the AI's summary.

It ends in an explicit sign-off block that does not close until every applicable check is
PASS (or has a documented justification) and **a named human statistician approves** model
choice, assumptions, and the headline claim, recording name, date, and
"approved / approved-with-caveats / rejected."

---

## The through-line

Climb only as high as the task needs. Always **use existing SOTA**. Decompose, specify the
data + question + constraints, demand a baseline and a skill score, ask for uncertainty, see
the validation design before the numbers, and keep the human sign-off. The art is in
*telling the agent what to do* — and in the rigor checkpoints you refuse to skip. AI
accelerates the work; you own the conclusion.

### Pointers
- The ladder, in full: [`docs/zero_to_hero.md`](zero_to_hero.md)
- Driving the L4 loop yourself: [`docs/TUTORIAL_autoresearch.md`](TUTORIAL_autoresearch.md)
- The protocol + gates: [`docs/autoresearch_protocol.md`](autoresearch_protocol.md)
- Rigor checklist: [`skills/env-stats-validate/SKILL.md`](../skills/env-stats-validate/SKILL.md)
- Flagship result: [`experiments/05_autoresearch_climate/README.md`](../experiments/05_autoresearch_climate/README.md)
