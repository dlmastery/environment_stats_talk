---
name: env-stats-validate
description: >
  Human-in-the-loop STATISTICAL RIGOR checklist for environmental statistics — the
  credibility backbone. Use when the user says "/env-stats-validate", "review this
  AI-produced result for rigor", "is this analysis trustworthy", "check for data
  leakage / multiple testing / uncertainty", or before reporting/publishing any result
  an AI or agentic loop produced. It walks an environmental statistician through the
  checks that must pass before a result is trusted, ending in an explicit
  "AI proposes, human decides" sign-off. AI accelerates; the human keeps final say.
version: 0.1.0
triggers:
  - "/env-stats-validate"
  - "validate this analysis"
  - "check for data leakage"
  - "is this result trustworthy"
  - "rigor checklist before publishing"
---

# env-stats-validate — statistical rigor checklist (human-in-the-loop)

This is the credibility backbone of the talk. An AI (or the `env-autoresearch-setup`
loop) can produce a result fast; **rigor is what makes it trustworthy.** Run this
checklist on any AI-produced environmental-statistics result before reporting it. The
framing throughout: **AI proposes, the human statistician decides** — you keep final
say on model choice, assumptions, multiple-testing, leakage, and uncertainty.

This skill is a checklist, not an automation. For each item: state the check, what to
look for, and how to gather evidence; mark **PASS / FAIL / N-A** with a one-line note
and a pointer to the evidence (plot/table/code). Do not sign off until every applicable
item is PASS or has a documented justification. The compact form is in
`references/checklist.md` (copy it into the report and fill it in).

> Motivation: prompting LLMs to do environmental/ecological statistics is only useful if
> the output holds up to the rigor the field demands. (Verify the Brown & Spillias 2026
> *Methods in Ecology & Evolution* reference before citing it on a slide — do not assert
> the exact title/date/venue until confirmed against `ledgers/CITATIONS-TO-VERIFY.md`.)

## The checks

### 1. Data leakage — temporal & spatial split correctness

The single most common way an impressive environmental result is wrong.

- **Temporal:** is the validation/test set strictly in the future of training? No
  shuffling of time series, no feature computed using future values, and a **purge +
  embargo** around split boundaries sized to the label horizon. Use a walk-forward split,
  not random k-fold, for forecasting (see `common/synthetic_climate.time_split` for the
  no-shuffle baseline).
- **Spatial:** are autocorrelated neighbors kept on the same side of the split? Hold out
  whole **basins / grid cells / stations / regions** together (spatial GroupKFold). A
  random pixel/point split leaks because nearby points are nearly identical.
- **Preprocessing leakage:** were scalers/imputers/feature selection/climatology baselines
  fit on **training only** (inside the CV fold), not on the full dataset?
- Look for: suspiciously high skill, near-perfect tail fits, or a model that beats a
  foundation-model zero-shot baseline by an implausible margin → suspect leakage first.

### 2. Stationarity & autocorrelation

- Is the series/residual stationary enough for the method used? Check for trend and
  changing variance (the synthetic climate data here has a warming trend and intensifying
  extremes by design). For trend detection use Mann-Kendall; for change points,
  change-point tests (verify the specific library).
- Are residuals autocorrelated? Inspect the ACF/PACF or run a test (e.g. Ljung-Box,
  Durbin-Watson). Autocorrelated residuals invalidate naive i.i.d. standard errors and
  inflate apparent significance — use block bootstrap / HAC SEs / explicit AR terms.

### 3. Distributional assumptions

- Does the method's distributional assumption match the data? Environmental data is
  rarely Gaussian: precipitation is intermittent and heavy-tailed; extremes follow GEV/GPD;
  occurrence/count data is discrete and over-dispersed; compositional data is constrained.
- For **EVT** specifically: justify the threshold (POT) or block size (block maxima),
  check the parameter-stability / mean-residual-life plot, and report the GEV/GPD shape
  parameter with its uncertainty (not a point estimate alone).
- Check transforms (log, Box-Cox, logit for compositions) and verify back-transformation
  is unbiased where it matters.

### 4. Multiple-testing correction

- How many hypotheses/grid-cells/species/stations were tested? Per-cell p < 0.05 across a
  large grid produces many false positives by construction.
- Apply and report a correction: Bonferroni (conservative) or FDR / Benjamini-Hochberg
  (typical for many spatial tests). State the family of tests and the corrected threshold.
- Watch for implicit multiplicity: an autoresearch loop that ran 50 experiments per
  backbone has effectively done massive multiple comparisons — the reported "best" is
  selection-biased. Confirm the winner on a held-out fold it was never selected on.

### 5. Uncertainty quantification (CIs / conformal)

- Are there calibrated uncertainty estimates, not just point predictions? Require
  confidence/prediction intervals or **conformal prediction** intervals, and show a
  coverage check (empirical coverage vs nominal level).
- For forecasts/fields: report ensemble spread or predictive intervals; for EVT, report
  return-level confidence intervals (bootstrap or profile-likelihood).
- A result with no uncertainty is not yet a scientific result.

### 6. Baseline / skill-score comparison

- Is the model compared against an honest baseline? For forecasting: climatology and
  persistence; for classification: majority-class / a simple Random-Forest-on-indices;
  for fields: latitude-weighted RMSE vs the same baseline.
- Report a **skill score** (`common/metrics.skill_score`): 1 = perfect, 0 = no better than
  the reference, < 0 = worse. "Low RMSE" alone is meaningless without the reference.
- Prefer the field-appropriate metric: latitude-weighted RMSE / ACC for gridded climate
  fields (`common/metrics.latitude_weighted_rmse`, `anomaly_correlation`).

### 7. Reproducibility (seeds / versions / data provenance)

- Are random seeds fixed and recorded? Can the result be regenerated bit-for-bit (or to
  documented tolerance) from the committed code + config?
- Are package versions and the data snapshot/provenance recorded (see the
  `climate-data-fetch` `.meta.json` sidecar)? Pin versions; note GPU/CPU determinism caveats.
- Can a reviewer run it with no API key on synthetic data (runs-anywhere) to sanity-check
  the pipeline logic?

### 8. AI-specific checks

- **No invented facts.** Verify every paper/statistic/date the AI cited; flag anything
  unconfirmed as `(verify)` and route it to `ledgers/CITATIONS-TO-VERIFY.md`. Never let a
  hallucinated citation reach a slide.
- **Goal integrity (Goodhart).** Confirm the composite metric was not silently rewritten
  mid-project (check the fingerprint from `env-autoresearch-setup`).
- **Prompt-induced bias.** Did the AI's framing/prompt steer the conclusion? Re-check the
  headline claim against the raw evidence, not the AI's summary.

## Sign-off — AI proposes, human decides

Do not report or publish until:

- [ ] Every applicable check above is **PASS** (or has a written, defensible justification).
- [ ] Leakage (temporal AND spatial) was actively ruled out, not assumed.
- [ ] An honest baseline + skill score is reported.
- [ ] Uncertainty is quantified and its coverage checked.
- [ ] Multiple testing is corrected (or shown not to apply).
- [ ] Reproducibility artifacts (seeds, versions, provenance) are committed.
- [ ] Every AI-supplied citation/number is verified or `(verify)`-flagged.
- [ ] **A named human statistician signs off** on model choice, assumptions, and the
      headline claim. Record: name, date, and "approved / approved-with-caveats / rejected".

The AI accelerates the work; the human owns the conclusion.
