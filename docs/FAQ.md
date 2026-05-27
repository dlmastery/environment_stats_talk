# FAQ — honest answers for a skeptical audience

The audience for this talk (and repo) is environmental statisticians — rigorous, busy,
and rightly skeptical of hype. These answers are written in that spirit: where the honest
answer is "no," it says no.

---

### Is this real data?

**No — every committed number comes from a synthetic generator, on purpose.** Each
experiment's generator (`common/synthetic_*.py`) produces *physically plausible* series
with the same schema as the real source it stands in for: ERA5-like temperature/
precipitation, Sentinel-2-like 5-band patches, CAMELS-like rainfall-runoff, OpenAQ-like
PM2.5 with weather drivers, etc. This is a deliberate **"runs-anywhere"** choice — the
results reproduce with no API keys, no network, and no GPU, so anyone can sanity-check the
*pipeline logic* in minutes. The synthetic numbers are **ceilings/illustrations, not
real-world performance**, and each generator injects the structure that makes the
before/after contrast real rather than rigged by noise. The exact swap to each real source
(ERA5, iNaturalist/GBIF, Sentinel-2, station+DEM, OpenAQ+reanalysis, CAMELS) is documented
in [`GUIDE_adapt_to_real_data.md`](GUIDE_adapt_to_real_data.md), and the fair-stand-in
reasoning and limitations are in [`ADDENDUM_methodology.md`](ADDENDUM_methodology.md).

### Does AI replace the statistician?

**No.** The framing throughout is **"AI proposes, the human statistician decides."** The
agent (Claude Code + an AI-for-science model) *proposes and benchmarks* — it writes the
pipeline, stands up baselines, runs the comparison, and adds rigor checks fast. The human
keeps final say on the **research question, assumptions, model choice, multiple-testing
discipline, leakage, uncertainty interpretation, and peer review** — and signs off by name.
That division of labour is encoded in the [`env-stats-validate`](../skills/env-stats-validate/SKILL.md)
skill, which ends in an explicit human sign-off and is not satisfied until leakage is ruled
out, an honest baseline and skill score are reported, uncertainty is quantified and its
coverage checked, multiple testing is corrected, and every AI-supplied citation is verified.

### Why does the classical method sometimes win?

**Because that *is* the point.** The verdicts across the nine experiments in
[`RESULTS.md`](../RESULTS.md) are mixed on purpose. **SARIMA beats the LSTM** at ≥7-day
lead in Exp01 (one-step daily temperature is a near-persistence problem); on easy land
cover **RF = CNN** (Exp04); and **kriging keeps a calibrated variance surface the RF
lacks** (Exp06), so regression-kriging — not the tree model — is the true hero. The agent's
value is letting you *find that out fast, fairly, and reproducibly* — including discovering
that the classical tool is already enough. A talk that claimed the neural net always wins
would lose this audience in the first counterexample; honesty about the ties and
classical-wins is what makes the genuine wins (hydrology +0.56 NSE, air quality −5.3 RMSE,
conformal ~7× tighter calibration, hard-mode land cover +0.36 acc) credible.

### How do I trust AI output?

Run the [`env-stats-validate`](../skills/env-stats-validate/SKILL.md) checklist on it —
that skill is the credibility backbone of the whole project. It walks through the eight
checks a result must pass before it is trusted: **(1) data leakage** (temporal *and*
spatial split correctness — the single most common way an impressive result is wrong),
**(2) stationarity & autocorrelation**, **(3) distributional assumptions**, **(4)
multiple-testing correction**, **(5) calibrated uncertainty with a coverage check**, **(6)
honest baseline + skill score**, **(7) reproducibility (seeds/versions/provenance)**, and
**(8) AI-specific checks** (no invented facts, Goodhart/goal integrity, prompt-induced
bias). Each item is marked PASS/FAIL/N-A with evidence; you do not sign off until every
applicable item passes or has a written, defensible justification.

### Is it reproducible?

**Yes.** Every generator uses **deterministic seeds**; the committed `results/` are
produced by each experiment's `run_before_after.py` (or `run_loop.py` for the flagship),
and the **fast unit tests** run via `python run_all_tests.py` (each experiment in its own
subprocess to avoid `before/`/`after/` name collisions). Package versions are pinned in
[`requirements.txt`](../requirements.txt). The runs-anywhere path is CPU-only and keyless,
so a reviewer can regenerate the headline results without a GPU or any account. GPU/CPU
determinism caveats are noted where they apply.

### What about the speculative 2026 claims?

They are **quarantined and never asserted as fact** until independently verified. The
source transcript was machine-generated and dense with future-dated, unfindable, or
round-number claims; any such item lives in
[`ledgers/CITATIONS-TO-VERIFY.md`](../ledgers/CITATIONS-TO-VERIFY.md) (the quarantine
ledger) and [`ledgers/ITEMS.md`](../ledgers/ITEMS.md) (the trust-tagged inventory). The
rule is strict: **nothing in the quarantine may appear on a slide, in a handout, in a
README, or in a code comment as TRUE until verified against a primary source** — and if a
talking point depends only on a quarantined claim, you cut the talking point rather than
weaken the rigor. Some items *have* since been verified (with corrections — e.g. the
AI-Scientist Nature paper is titled "Towards end-to-end automation of AI research," and
"Aletheia" runs on Gemini 3 *Deep Think*, not "Gemini 3.5"); others remain unverified and
are not used. Even numbers attached to *real* papers are treated as needs-verification
until re-derived from the primary source.

### Do I need a 4090?

**No.** The headline results are reproducible on **CPU with no GPU** — that is the
runs-anywhere rule. The RTX 4090 (the build machine) is only used to *scale* the AFTER
models (more data, bigger networks, longer context, foundation-model checkpoints) and to
produce a couple of larger committed runs (e.g. the hard-mode land-cover win in Exp04). The
device is auto-selected (`get_device()` / `pick_device()` returns CUDA when present and
falls back to CPU), so the same code runs either way without a flag. Several experiments
(02, 06, 07, 12) have **no GPU dependency at all**.

### "100× faster" — really?

**Defined precisely: it is about *person-time* and *rigor*, not a wall-clock multiplier.**
The repo does **not** assert a "100× faster" or "days vs years" figure as a measured fact —
those transcript-style multipliers are explicitly quarantined as unsourced marketing claims
(see [`ledgers/CITATIONS-TO-VERIFY.md`](../ledgers/CITATIONS-TO-VERIFY.md) §C4). What the
before/after slides actually quantify is the collapse in **human build effort**: tasks that
are typically a day-to-weeks of bespoke coding (build + benchmark four forecasters with skill
scores and plots; GEV + bootstrap CIs + multi-index trend + assumption checks; train +
evaluate a CNN + change map; run a disciplined gated model search) become **one command in
minutes**. The acceleration is the agent compressing the engineering and bookkeeping *and
adding rigor gates a rushed human often skips* — while the human keeps every scientific
decision. The person-time figures in the experiment READMEs are labelled **illustrative
estimates of build effort, not measured benchmarks**; the only hard numbers claimed are the
repo's own committed metrics.
