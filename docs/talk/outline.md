# Talk Outline & Speaker Notes
### From Data to Discovery — Transforming Environmental Statistics with Claude Code & AI-for-Science

**Event:** International Environment Statistics Institute conference, Mexico City, 7–11 December 2026
**Session:** *"Modern Approaches to Environmental Statistics."*
**Audience:** Environmental statisticians / environmetricians — rigorous, busy, hype-skeptical, value statistical validity and concrete takeaways.
**Spine:** BEFORE (traditional environmetrics) vs AFTER (Claude Code + AI-for-science), with live demos drawn from this repo's experiments.

**Conventions for this deck**
- Every slide carries exactly **one bold takeaway**.
- Only **verifiable anchors** are named (arXiv IDs + canonical datasets). Measured numbers come from this repo's `RESULTS.md` (committed after experiment runs) — on slides they appear as `[see RESULTS.md]`.
- Speculative / future-dated claims are **never** asserted; where the audience may expect a number we have not measured, we write `(verify)`.
- Target length: ~16 slides for a ~30–35 min slot, leaving room for the live demo and Q&A.

---

## Slide 1 — Title: From Data to Discovery

- *From Data to Discovery: Transforming Environmental Statistics with Claude Code & AI-for-Science*
- Presenter name, affiliation, contact (eranti@gmail.com)
- Session: "Modern Approaches to Environmental Statistics," Mexico City, Dec 2026
- One-line thesis: same scientific question, BEFORE vs AFTER
- "Every claim today points to runnable code with committed results."

**Visual:** Split-screen title image — left a hand-drawn time-series on graph paper (BEFORE), right a terminal + forecast plot (AFTER); muted grey vs blue palette matching the repo (`#9aa0a6` / `#1a73e8`).

**Speaker notes:** Welcome — I'm going to make one argument and back it with receipts. The argument is that the daily work of an environmental statistician is being reshaped by AI coding agents plus AI-for-science models, and the honest way to show that is BEFORE versus AFTER on tasks you already do. Everything on these slides is backed by a companion repository with runnable code and committed numbers, so nothing here is a demo-day illusion.

**TAKEAWAY:** **This is a working talk: every claim has runnable code behind it.**

---

## Slide 2 — Why now

- Three curves crossed: capable coding agents, AI-for-science models, commodity GPUs (a single RTX 4090 ~16 GB)
- The bottleneck moved from *compute* and *code* to *asking the right scientific question*
- Verifiable signals this is real: ClimateLLM (arXiv:2502.11059), CLLMate (arXiv:2409.19058), EagleVision (arXiv:2503.23330)
- Open data is abundant: ERA5, GBIF, Sentinel-2, CAMELS, OpenAQ
- Not "AI replaces the statistician" — "the statistician now drives a team of agents"

**Visual:** Three overlapping curves (agent capability / AI-for-science models / affordable GPU) crossing a "good enough for daily research" line; no fabricated y-axis values — qualitative shape only, labeled "schematic."

**Speaker notes:** Why is this a 2026 conversation and not a 2020 one? Three things matured at once: coding agents that can actually run and debug your pipeline, domain models trained for science, and GPUs cheap enough to sit under your desk. The consequence is that the rate-limiting step is no longer writing code or finding compute — it is posing a well-formed scientific question and validating the answer, which is exactly the statistician's strength.

**TAKEAWAY:** **The bottleneck moved from code to questions — and that favors statisticians.**

---

## Slide 3 — The field map: a taxonomy of environmental-statistics tasks

- Forecasting (climate/weather time series) → Exp01
- Extremes & trend detection (GEV/POT, Mann-Kendall) → Exp02
- Information extraction from text (events, species interactions) → Exp03
- Remote sensing / land cover (spatial classification) → Exp04
- Cross-cutting: spatial stats, UQ, Bayesian hierarchical, source apportionment (backlog)

**Visual:** A 5-cell taxonomy grid; each cell carries its experiment number and a tiny icon, with a column on the right reading "where AI/Claude Code fits." Reference `docs/taxonomy.md`.

**Speaker notes:** Before we talk tooling, here's the map of the work. I've grouped environmental-statistics tasks into a small taxonomy, and the companion repo has a paired experiment for the first four cells plus a backlog for the rest. The point of the map is honesty about scope: AI helps unevenly across these cells, and I'll show you where it helps a lot and where the human still carries the load.

**TAKEAWAY:** **Map the work first; AI's leverage differs cell by cell.**

---

## Slide 4 — The BEFORE/AFTER paradigm

- BEFORE: one person, specialist tools, glue scripts, days-to-weeks per study
- AFTER: you brief an agent; it writes code, runs it, plots, and reports — minutes-to-hours
- The unit of comparison: **same question, same data, same metric** — only the workflow changes
- What is preserved: the statistics (assumptions, splits, multiple testing, uncertainty)
- What changes: iteration speed and the breadth of what you can try

**Visual:** Two swim-lanes (BEFORE grey / AFTER blue) for the same task, each with its step sequence; the AFTER lane is visibly shorter. Pull from `docs/before_after.md`.

**Speaker notes:** Here's the framing for the whole talk. For each task I hold the science fixed — same dataset, same metric, same validation — and change only the workflow. BEFORE is the careful artisanal pipeline we all know. AFTER is briefing a coding agent that writes, runs, and reports. The claim is not "better statistics," it's "the same rigor, dramatically faster iteration," and I'll be explicit when AFTER is merely faster versus actually better.

**TAKEAWAY:** **Hold the science fixed; change only the workflow — then measure.**

---

## Slide 5 — Does "faster" mean "sloppier"? Rigor is the constraint

- Speed without validity is worthless to this audience — so rigor is a hard gate, not an afterthought
- The repo encodes a statistical-rigor checklist: leakage, train/val/test discipline, multiple testing, uncertainty
- Chronological splits for time series (never leak the future): `common.synthetic_climate.time_split`
- Latitude-weighted RMSE / ACC for gridded fields: `common.metrics`
- The human signs off on assumptions; the agent does the labor

**Visual:** A "rigor checklist" card (leakage / splits / multiplicity / UQ / reproducibility) stamped over the AFTER lane; cite the `env-stats-validate` skill.

**Speaker notes:** I want to address the obvious objection up front, because you're statisticians. Faster is only interesting if it's still valid. So in this repo rigor is a gate the agent must pass, not a nicety — chronological splits so we never leak the future, latitude weighting for gridded skill scores, explicit handling of multiple testing, and uncertainty on every estimate. The division of labor is simple: the agent does the typing and running; the human owns the assumptions.

**TAKEAWAY:** **Speed is only credible when rigor is a hard gate.**

---

## Slide 6 — Zero to hero: the Claude Code ladder

- L1 — Assistant: ask for a function, a plot, a sanity check
- L2 — Pipeline builder: "fetch the data, fit the model, validate, plot, report"
- L3 — Skilled operator: reusable **skills** (data-fetch, stats-validate) + project conventions
- L4 — Autoresearcher: the agent runs a multi-experiment loop and keeps a research journal (Exp05)
- You can start at L1 on Monday and climb at your own pace

**Visual:** A 4-rung ladder (L1→L4) with the matching repo artifact beside each rung; reference `docs/zero_to_hero.md`.

**Speaker notes:** "Zero to hero" is a ladder, not a leap. Level one is using the agent as a smart assistant for a single function or plot. Level two is handing it an entire pipeline. Level three is when you teach it reusable skills and your conventions so it behaves like a trained lab member. Level four is autonomous research, which I'll demo near the end. The reassuring part is that level one pays off on day one.

**TAKEAWAY:** **It's a ladder you can climb gradually — L1 helps on day one.**

---

## Slide 7 — Live demo (1): Climate time-series forecasting — Exp01

- Task: forecast a daily temperature-like series; metric RMSE and anomaly correlation (ACC)
- BEFORE: persistence / ARIMA baseline (traditional)
- AFTER: learned model (LSTM/TCN on the 4090) + foundation-model zero-shot variant
- Command: `python experiments/01_climate_timeseries_forecast/run_before_after.py`
- Numbers + forecast plot: `[see RESULTS.md]`

**Visual:** Live terminal, then the generated forecast plot + a BEFORE/AFTER RMSE bar (the repo's `before_after_bars` helper). Recorded-backup screenshot on the slide as fallback.

**Speaker notes:** Let's make it concrete. This experiment forecasts a daily temperature series and scores it with RMSE and anomaly correlation, the standard skill metric in numerical weather prediction. I run one command; it executes both the traditional baseline and the learned model, then writes the metrics and a plot. Watch the time it takes — and note that the validation split is chronological so we are not cheating by peeking at the future. The exact numbers are in RESULTS.md so I never quote a figure I haven't measured.

**TAKEAWAY:** **One command reproduces the baseline and the learned forecast — with honest splits.**

---

## Slide 8 — Live demo (2): Extremes & trends — Exp02

- Task: trend detection + return levels on a precipitation-like series
- BEFORE: hand-coded Mann-Kendall + GEV fit, manual diagnostics
- AFTER: Claude Code pipeline that fits, validates, and emits CIs + a rigor checklist
- Honest framing: the *estimator* is the same; the *workflow* and *uncertainty reporting* improve
- Artifacts: return-level plot + trend confidence intervals `[see RESULTS.md]`

**Visual:** Return-level curve with confidence band + a trend slope with its CI; small inset of the auto-generated rigor checklist.

**Speaker notes:** Extremes are where statisticians are rightly protective, so here the estimator is unchanged — it's still Mann-Kendall and a GEV fit. What changes is that the agent assembles the pipeline, attaches confidence intervals, and runs the rigor checklist automatically, including a note on multiple testing if we scan many stations. The lesson is that AFTER doesn't reinvent extreme-value theory; it removes the glue-code tax and makes the uncertainty reporting non-optional.

**TAKEAWAY:** **For extremes, AI removes glue-code tax — the EVT stays exactly as rigorous.**

---

## Slide 9 — Live demo (3): Biodiversity from text — Exp03

- Task: extract species interactions from free text into a structured network
- BEFORE: regex / keyword extraction — brittle, low recall on phrasing variation
- AFTER: structured-LLM extraction → interaction graph; scored by precision/recall
- Grounding datasets named only as canonical sources: GBIF, iNaturalist
- Artifacts: precision/recall table + interaction network graph `[see RESULTS.md]`

**Visual:** Side-by-side: a sparse regex-derived graph vs a richer LLM-derived interaction network; precision/recall bars beneath.

**Speaker notes:** This is the cell where AFTER is not just faster but genuinely better. Pulling species interactions out of unstructured text with regex is brittle and misses paraphrase. A structured-LLM extraction recovers far more of the signal — but, crucially, we score it with precision and recall against a labeled set rather than trusting it. So even here, the believable version of the result is the one with a confusion-matrix-grade evaluation attached.

**TAKEAWAY:** **For text→structure, AI is genuinely better — but only with precision/recall attached.**

---

## Slide 10 — Live demo (4): Remote-sensing land cover — Exp04

- Task: land-cover classification from spectral features
- BEFORE: Random Forest on spectral indices (NDVI etc.)
- AFTER: CNN on the 4090 (+ pretrained embeddings); accuracy / macro-F1 + change map
- Verifiable anchor for the modality: EagleVision (arXiv:2503.23330); data source: Sentinel-2
- Artifacts: accuracy/F1 table + change map `[see RESULTS.md]`

**Visual:** A classified land-cover map and a change-detection map side by side; accuracy/F1 comparison bar.

**Speaker notes:** Remote sensing is the most compute-flavored cell, and it's where the desktop GPU earns its keep. The traditional move is a random forest on spectral indices; the AFTER move is a small CNN with pretrained embeddings, trained on the 4090 in minutes. I report accuracy and macro-F1, not just overall accuracy, because class imbalance is the norm in land cover. EagleVision is the kind of verifiable work showing this modality is maturing fast.

**TAKEAWAY:** **A desktop GPU now trains real remote-sensing models in minutes.**

---

## Slide 11 — The flagship: an autonomous research loop — Exp05

- The agent *is* the researcher: Diagnose → Cite → Hypothesize → Predict → Execute → Analyze → Checkpoint
- Hard gates before any run: citation-rigor + reasoning-completeness (no ungrounded experiments)
- Pluggable backbones / splits / composite metric; champion archive + research journal
- Adapted from the user's own `dlmastery/autoresearch` (`generalized_ml_autoresearch`)
- Runs unattended on a single 4090

**Visual:** The 7-step loop as a circle with the two hard gates drawn as locks before "Execute"; a thumbnail of the champion-archive folder + research-journal entry. Reference `docs/autoresearch_protocol.md`.

**Speaker notes:** This is the part that feels like science fiction but isn't. The autoresearch loop turns Claude Code into the researcher: it diagnoses the current best model, cites prior work, forms a hypothesis, predicts the outcome, runs exactly one experiment, analyzes it, and checkpoints. Two gates matter for this audience — it cannot run an experiment until its citations and its reasoning pass a completeness check, which is the machine-enforced version of "don't fish." It produces a champion archive and a human-readable journal you can audit afterward.

**TAKEAWAY:** **The loop makes the agent a researcher — gated so it can't fish for significance.**

---

## Slide 12 — Autoresearch, honestly: what it does and doesn't do

- DOES: explore a defined search space fast, document every step, archive the winner
- DOESN'T: choose your scientific question, your loss, or your validity criteria — you do
- Guardrails: composite-metric fingerprint detects mid-project goal rewrites
- Reproducibility: deterministic seeds; crash-recovery checkpoints
- Output is a draft to *review*, not a paper to *trust* blindly

**Visual:** A two-column "DOES / DOESN'T" card; a small lock icon for the metric-fingerprint guardrail.

**Speaker notes:** Let me be careful here, because over-claiming would lose this room. The loop is excellent at exhaustively and transparently searching a space you define and archiving what wins. It does not pick the question, the loss function, or the standard of validity — that is your job, and the system is designed to keep it your job, including a fingerprint that flags if the objective quietly drifts mid-run. Treat its output as a well-documented draft to review, never as an oracle.

**TAKEAWAY:** **Autoresearch searches the space you define — it never defines the question.**

---

## Slide 13 — Challenges & rigor: keeping the human in the loop

- Leakage & look-ahead: the most common failure — enforce chronological/spatial splits
- Multiple testing across many stations/species/cells — correct for it
- Uncertainty is mandatory: report intervals, not point estimates
- Hallucinated citations / numbers — verify; quarantine the unverified (`ledgers/CITATIONS-TO-VERIFY.md`)
- Reproducibility: seeds, committed results, runs-anywhere on synthetic data

**Visual:** A "human-in-the-loop" diagram: agent proposes → human checks the five risk items → accept/reject; the five items listed as a checklist.

**Speaker notes:** Here are the failure modes I worry about, and how this setup defends against each. Leakage is the big one, so splits are chronological or spatial by construction. Multiplicity is corrected explicitly when we scan many units. Uncertainty is reported as intervals, never bare point estimates. And because LLMs can fabricate references, anything not independently verified is quarantined and never stated as fact — that discipline is why I keep saying "see RESULTS.md" instead of quoting numbers from memory.

**TAKEAWAY:** **The human owns five checks — leakage, multiplicity, uncertainty, citations, reproducibility.**

---

## Slide 14 — What you can do Monday

- Today: install Claude Code; ask it to reproduce one figure from your last paper
- This week: hand it a full small pipeline — "fetch ERA5, fit a trend, validate, plot"
- This month: write one **skill** that encodes your group's validation conventions
- Start on synthetic / small public data — no keys, no GPU — then scale up
- Clone this repo and run `experiments/01_.../run_before_after.py` as your template

**Visual:** A three-step "Today / This week / This month" timeline card mirroring the handout; a small QR placeholder to the repo.

**Speaker notes:** Concrete next steps, because I want you to leave with momentum. Today, just install the agent and ask it to redraw one figure from your most recent paper — low stakes, instant intuition. This week, give it a whole small pipeline end to end. This month, encode your group's validation rules as a reusable skill so the agent inherits your standards. Start on synthetic or small public data so nothing blocks you, then scale to real data and the GPU.

**TAKEAWAY:** **You can start today on synthetic data — no keys, no GPU required.**

---

## Slide 15 — Vision: the 100×-faster lab

- The scarce resource becomes good questions and good judgment, not labor
- Each scientist runs a small team of agents; reproducibility and audit trails improve, not degrade
- Open data + desktop GPUs + agents → individual scientists doing institution-scale work
- The statistician's role grows: assumptions, validity, interpretation, communication
- The repo is the proof-of-concept; your lab is the next instance

**Visual:** A single scientist at a laptop with several "agent" worker icons fanning out to the taxonomy cells; calm, optimistic framing.

**Speaker notes:** Zoom out. If labor stops being the constraint, then judgment becomes the differentiator — and judgment is what statistical training produces. I think the near future is each of us orchestrating a handful of agents across the taxonomy, with better audit trails than we have today because everything is logged and reproducible. The role doesn't shrink; it moves up the stack to the parts that were always the science.

**TAKEAWAY:** **When labor is cheap, scientific judgment becomes the differentiator.**

---

## Slide 16 — Thank you / resources / Q&A

- Repo: `environment_stats_talk` (companion code + committed results + this deck)
- Verifiable anchors: ClimateLLM 2502.11059 · CLLMate 2409.19058 · AI co-scientist 2502.18864 · AI-Scientist-v2 2504.08066 · EagleVision 2503.23330
- Datasets: ERA5 · GBIF · Sentinel-2 · CAMELS · OpenAQ
- Handout: `docs/talk/handout.md` (copy-paste prompts + action plan)
- Contact: eranti@gmail.com — questions welcome

**Visual:** A clean resource card with the arXiv IDs and dataset names, a QR placeholder to the repo, and contact info.

**Speaker notes:** Thank you. Everything I showed lives in the companion repo, including this deck and a one-page handout with copy-paste prompts you can try tonight. The papers on this slide are the verifiable anchors I built on — arXiv IDs so you can check them yourself — and the datasets are the standard open sources in our field. I'm happy to take questions, and I'm especially interested in where you think the human-in-the-loop checks need to be stronger.

**TAKEAWAY:** **Take the handout, clone the repo, and try one prompt tonight.**

---

### Citation & honesty appendix (not a slide)
- **Verified anchors only on slides:** ClimateLLM (arXiv:2502.11059), CLLMate (arXiv:2409.19058), "Towards an AI co-scientist" (arXiv:2502.18864), AI-Scientist-v2 (arXiv:2504.08066), EagleVision (arXiv:2503.23330). Confirm exact titles/versions/dates against the arXiv pages before presenting.
- **Datasets (canonical, safe to name):** ERA5, GBIF, iNaturalist, Sentinel-2, CAMELS, OpenAQ.
- **Never assert (per `ledgers/CITATIONS-TO-VERIFY.md`):** any "Nature, May 2026" dating, "OpenAI Erdős" breakthrough, "GPT-5.4 / Gemini 3.5", "WeatherNext 2" specifics, or any invented statistic. Use `(verify)` or point to `RESULTS.md`.
- **Numbers:** all measured metrics live in `RESULTS.md` (committed after experiment runs). Slides reference them as `[see RESULTS.md]` rather than hard-coding figures that could drift.
