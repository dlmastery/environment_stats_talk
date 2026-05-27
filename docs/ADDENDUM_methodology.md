# Methodology addendum — synthetic data, honest verdicts, and threats to validity

This addendum is for the reader who asks the right hard questions: *Why synthetic data? Is
it rigged? What can these numbers actually support? How do I know the citations are real?*
It is the companion to the [`FAQ`](FAQ.md), the data-swap
[`GUIDE`](GUIDE_adapt_to_real_data.md), and the rigor checklist in
[`env-stats-validate`](../skills/env-stats-validate/SKILL.md).

---

## 1. Why synthetic data

The repo's headline rule is **runs-anywhere**: every committed result must reproduce from
synthetic + small public data with **no API keys, no network, and no GPU**, using
deterministic seeds (see [`CONTRIBUTING.md`](../CONTRIBUTING.md) and
[`CLAUDE.md`](../CLAUDE.md)). Synthetic data is what makes that possible, and it buys four
things:

- **Reproducibility for a reviewer.** Anyone — no ECMWF account, no Sentinel Hub
  credentials, no 4090 — can regenerate the numbers in minutes and sanity-check the
  *pipeline logic* (splits, leakage handling, metric computation) before investing in a real
  fetch.
- **A known ground truth.** With a synthetic field we know the true return level, the true
  spatial field, the true interaction labels, the true change mask. That lets us measure
  error against truth (e.g. Exp06 scores against the noise-free field, not noisy
  re-observations) and verify a method recovers the *injected* signal.
- **An honest, controlled before/after.** Each generator injects exactly the structure that
  separates BEFORE from AFTER, so the contrast reflects a real mechanism rather than a noise
  artifact (see §2).
- **Speed and determinism for CI.** The unit tests run on CPU in seconds and assert claims,
  not just "code runs."

The cost is stated plainly everywhere: **synthetic numbers are ceilings/illustrations, not
real-world performance.** They demonstrate the *workflow* and the *mechanism*; the
real-world number requires the real-data swap in the [`GUIDE`](GUIDE_adapt_to_real_data.md).

## 2. How each generator is a fair stand-in (not rigged)

A "fair stand-in" means the generator embeds the same physical/statistical structure as the
real source — including the structure that makes the BEFORE method fall short — without
hand-tuning the outcome. The signal is injected by mechanism; the verdict is then *measured*,
not assumed. By experiment:

- **Exp01/05 — temperature forecasting (`daily_temperature`).** Annual + semi-annual
  seasonal cycle, multidecadal warming trend, **AR(1) weather persistence**, and sporadic
  heat extremes. The AR(1) persistence is *why* one-step forecasting is near-persistence and
  the LSTM only modestly beats the naive floor — the generator does not flatter the neural
  net; it reproduces the real difficulty (SARIMA even wins at longer leads).
- **Exp02 — extremes (`daily_precipitation`).** Intermittent (many dry days), seasonally
  modulated wet-day probability, and a **heavy upper tail whose scale intensifies over
  time**. That intensification is the trend both workflows must recover; it also creates the
  non-stationarity the `validate()` gate flags.
- **Exp03 — text extraction (`citizen_comments`/`gold_interactions`).** ~35% pure-distractor
  comments, plus deliberately varied phrasing (passive voice, synonyms, indirect/figurative
  forms). The varied phrasing is *why* a literal verb list under-recalls — the generator
  mirrors how messy real citizen-science text is, then measures the gap.
- **Exp04 — land cover (`multispectral_patches`, `multispectral_patches_hard`).** Per-class
  spectral signatures + spatial texture in the real Sentinel-2 `(N,5,H,W)` reflectance
  contract. The default classes are *well-separated* (so RF-on-indices honestly near-ties the
  CNN); **hard mode** makes two class pairs share a near-identical mean spectrum and differ
  only in **oriented vs isotropic texture** — structurally invisible to NDVI/NDWI, readable
  by a CNN. (The module note documents that a smooth-vs-speckle pair was *avoided* precisely
  because it would leak an NDVI-variance tell a tree could exploit — i.e. the generator is
  designed *not* to rig the win.)
- **Exp06 — spatial (`synthetic_spatial_field`).** `z = trend + lapse·elevation + GRF`: a
  smooth autocorrelated part kriging is *built to model well*, plus a short-scale
  covariate-driven part coordinate-only kriging *structurally cannot* follow. Kriging is a
  strong baseline (it clearly beats IDW); the AFTER win is exactly the covariate gap, and the
  generator's tests assert the covariate is genuinely informative and varies on shorter
  scales than the GRF.
- **Exp07 — air quality (`synthetic_pm25`).** `emission(hour,dow,season) · ventilation(wind)
  · dilution(boundary_layer) · (1 + boost·episode)` with only modest residual self-
  persistence — so the spikes are genuinely weather-driven, not raw stickiness. The committed
  **history-only ablation** proves the win comes from the meteorological covariates, not the
  model class.
- **Exp08 — hydrology (`synthetic_catchment`).** A minimal conceptual bucket (HBV/GR4J in
  spirit): nonlinear saturation-excess runoff `(S/Sₘₐₓ)^β`, a degree-day snow module, slow
  baseflow, and a routing kernel. The nonlinear, state-dependent memory is precisely what a
  linear regression cannot represent and an LSTM's cell state can — the textbook reason
  LSTM-rainfall-runoff wins.
- **Exp12 — uncertainty.** Reuses `daily_temperature` in a deliberately **heteroscedastic,
  heavy-tailed** regime (real `heat_extreme_rate` / `warming_c_per_decade` knobs, measured
  excess kurtosis ≈ 5–7), which is exactly where a single Gaussian σ̂ mis-sizes the band and
  the distribution-free conformal interval earns its keep.

In every case the generator's *mechanism* is documented in the module docstring and a
`REAL_DATA_NOTE`, and the schema matches the real source so the swap is drop-in.

## 3. The honest-verdict philosophy

The verdicts in [`RESULTS.md`](../RESULTS.md) are **mixed on purpose**, and that is the
scientific spine of the talk. "AFTER" means *Claude Code + an AI-for-science model wrote,
ran, and benchmarked the workflow* — the headline is the collapse in **human effort** and
the **breadth of rigor added**, not a claim that a neural net always beats classical
statistics. So the repo reports:

- **Clean AFTER wins** where structure rewards them (hydrology +0.56 NSE, air quality −5.3
  RMSE, conformal ~7× tighter calibration, hard-mode land cover +0.36 acc, spatial
  interpolation −0.70 RMSE, biodiversity recall).
- **A rigor/uncertainty win** (extremes — same question, far more honest uncertainty).
- **Honest ties and classical-wins** where they belong: **SARIMA beats the LSTM** at ≥7-day
  lead; easy land cover RF = CNN; and **kriging keeps a calibrated variance surface the RF
  lacks**, so regression-kriging is the true hero.

We never manufacture a synthetic win the data does not support. Where the cheap baseline is
already excellent, the README says the AFTER value is *capability/effort*, not accuracy. This
honesty is what lets a skeptical audience trust the genuine wins.

## 4. Limitations / threats to validity

State these openly; they are the questions the audience will ask.

- **Synthetic ceilings.** A closed, generated world has a ceiling a real one does not. Exp03's
  1.00 P/R/F1 is the ceiling of a *closed-vocabulary* corpus — real text will score lower and
  noisier, which is exactly why human verification of extracted triples is mandatory. Treat
  every synthetic number as an illustration of the mechanism, not a forecast of field
  performance.
- **Exchangeability.** Conformal prediction's coverage guarantee (Exp12) assumes exchangeable
  data, but forecast residuals are weakly **serially dependent**, which breaks strict
  exchangeability. The experiment mitigates this (a calibration block adjacent to the test
  block) and **verifies empirical coverage on a held-out set** rather than trusting the
  guarantee blindly; for strongly non-exchangeable/shifting series it points to
  time-series-aware (weighted/adaptive) conformal.
- **Stationarity.** Several methods assume stationarity that real, warming-climate data
  violates. The Exp02 `validate()` gate flags a significant Mann-Kendall trend in the block
  maxima as a **stationary-GEV violation** (the return levels are then a record-average, not a
  current-climate estimate) and warns about lag-1 autocorrelation inflating Mann-Kendall
  false positives. The real-data swaps inherit non-stationarity the synthetic ceiling may
  understate.
- **Leakage realism.** Synthetic data lets us *demonstrate* leakage-safe splits (walk-forward,
  spatial GroupKFold, super-fold), but real data has subtler leakage paths (preprocessing
  fit on the full set, autocorrelated neighbours, look-ahead features). Run
  `env-stats-validate` #1 on real data; do not assume the synthetic discipline transfers for
  free.
- **Multiple testing.** Testing many indices/cells/species inflates family-wise error; the
  autoresearch loop running many experiments is itself massive multiple comparison, so the
  "best" is selection-biased and must be confirmed on a fold it was never selected on
  (`env-stats-validate` #4).
- **Person-time figures are estimates.** The "by hand vs with Claude Code" effort numbers in
  the READMEs are **illustrative estimates of build effort, not measured benchmarks** — they
  convey the acceleration, not a controlled study. The only hard numbers claimed are the
  repo's own committed metrics; transcript-style multipliers ("100× faster", "days vs years")
  are quarantined as unsourced (see §5).

## 5. The citation-verification stance

The audience punishes over-claiming, so the project runs a **zero-hallucination discipline**:

- Build only on **verifiable** papers/tools/datasets. Anything speculative, future-dated, or
  round-number is routed to [`ledgers/CITATIONS-TO-VERIFY.md`](../ledgers/CITATIONS-TO-VERIFY.md)
  (the quarantine ledger) and tagged in [`ledgers/ITEMS.md`](../ledgers/ITEMS.md) as
  `[VERIFIED]` / `[NEEDS-VERIFICATION]` / `[FLAGGED]`.
- **The quarantine rule is absolute:** nothing in the ledger may appear on a slide, in a
  handout, in a README, or in a code comment **as true** until verified against a primary
  source. If a talking point depends only on a quarantined claim, **cut the talking point**.
- **Even numbers on real papers are needs-verification.** A paper can be a verified anchor
  while its quoted metrics remain unverified until re-derived from the abstract/tables (a
  machine-generated transcript's numbers are never quoted directly).
- **Prefer the conservative anchor.** If only a preprint is confirmed, cite the preprint and
  say so; never silently upgrade a preprint to a journal paper. Where verification has been
  done, corrections are recorded (e.g. the AI-Scientist Nature paper's actual title is
  "Towards end-to-end automation of AI research"; "Aletheia" runs on **Gemini 3 Deep Think**,
  not "Gemini 3.5"; the Brown & Spillias paper is a 2026 *Methods in Ecology and Evolution*
  article with a DOI, not the transcript's guessed arXiv ID).
- In code and docs, anything not independently confirmed is marked **(verify)** rather than
  asserted.

## 6. Reproducibility & determinism

- **Seeds.** Every generator takes a seed and is deterministic; the experiments commit
  results from fixed-seed runs and the tests assert determinism where it matters.
- **Provenance.** `results/metrics.json` records config/provenance (seed, sizes, device,
  flags); real fetches additionally write a `.meta.json` sidecar (source, query, extent,
  time range, tool version, fetched-at, license, checksum) per the
  [`climate-data-fetch`](../skills/climate-data-fetch/SKILL.md) skill.
- **Pinned environment.** Core versions are pinned in [`requirements.txt`](../requirements.txt)
  (with notes on numpy-2.x ABI breaks in older statsmodels/matplotlib). Real-data/foundation
  dependencies are commented out — the committed results never require them.
- **Per-component testing.** [`run_all_tests.py`](../run_all_tests.py) runs each component in
  its own subprocess to avoid `sys.modules` collisions from the repeated `before/`/`after/`
  package names; this mirrors how the experiments are meant to be used (one at a time, from
  the repo root).
- **Determinism caveats.** GPU/CPU floating-point determinism is not bit-for-bit identical
  across devices; where a GPU "headline" run differs from the CPU smoke run, the difference is
  data/epoch scale, not a change in logic — the same code runs on either device via the
  auto-device selector.
