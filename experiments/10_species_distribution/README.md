# Experiment 10 — Species distribution modelling: BEFORE vs AFTER

Predict where a species is likely to occur from environmental covariates
(**temperature**, **precipitation**, **elevation**) given a presence-only sample
plus a uniformly drawn background / pseudo-absence sample — the standard SDM
setting in ecology and biogeography.

Everything here **runs anywhere**: synthetic data with deterministic seeds, no
API keys, no GPU required. CPU-only, sklearn-only, finishes in a few seconds.

---

## Why ML wins here (the scientific point)

Realistic species niches are **nonlinear and non-additive in climate**:

- A species has a **preferred temperature** and a **preferred precipitation**;
  suitability falls off in both directions (a Gaussian-style niche in (T, P)).
- The width of the precipitation niche **depends on elevation** — high-elevation
  sites tolerate a narrower precipitation band.
- The temperature optimum itself **shifts with elevation** (a lapse-rate
  analogue — cooler ridges, warmer valleys).

A **logistic regression with only linear and squared terms in (T, P, E)** — the
classical SDM workhorse, equivalent in spirit to a baseline MaxEnt with linear
+ quadratic features — can recover a single, separable Gaussian niche but
**cannot represent the elevation interactions** without the analyst manually
adding `T * E`, `T^2 * E`, `P^2 * E` (and getting their signs and scales right).
That manual feature crafting is precisely the slow, judgement-heavy step in
classical SDM workflows.

A **gradient-boosted tree ensemble** on the same three covariates learns those
interactions *automatically*: each split conditions on one variable given the
others, so the model recovers the elevation-dependent width and optimum from
the data without being told. The synthetic generator
(`common/synthetic_sdm.synthetic_sdm_dataset`) bakes in exactly this structure
— the BEFORE/AFTER gap is real, not rigged by noise.

---

## The story

**BEFORE — classical SDM via logistic GLM** (`before/glm_logistic.py`, pure CPU):

- Logistic regression on the three covariates plus a couple of squared terms
  (`temperature^2`, `precipitation^2`) — the textbook "linear + quadratic"
  niche baseline, fit with standardized inputs and mild L2 (`C=1.0`). Returns
  calibrated probabilities by construction of the logit link.

**AFTER — gradient-boosted-tree SDM** (`after/gbm_sdm.py`, pure CPU):

- `sklearn.ensemble.GradientBoostingClassifier` on the **same** three
  covariates (no hand-crafted interactions), wrapped in
  `CalibratedClassifierCV(method="isotonic", cv=3)` — a standard SDM hygiene
  step so the boosted-tree probabilities are properly calibrated (raw boosted
  trees rank well but can be over-confident; isotonic calibration preserves
  the ranking while fixing the probabilities).
- Modest configuration (n_estimators=200–300, max_depth=3, learning_rate=0.05).
  Not tuned per dataset.

**Metrics** (the SDM trio): **AUC** (higher is better, 1 = perfect ranking,
0.5 = random), **Brier score** (lower is better, 0 = perfect probability), and
**Pearson correlation between predicted P(presence) and the true noise-free
suitability** on the evaluation grid (only available because the dataset is
synthetic — that's the point: with the truth in hand, we can score map
quality directly, not just point-classification quality).

---

## Comparison table

Person-time is the realistic hand-build effort for an ecologist /
environmental statistician vs. the same artifact produced *with Claude Code*.
Metric cells reflect the committed run in `results/`; see `results/summary.md`
for the exact numbers from the latest run on this machine.

| Method | Person-time (by hand) | Person-time (with Claude Code) | Wall-time (run) | AUC | Brier | Artifacts |
|---|---|---|---|---|---|---|
| Before: GLM (logit + sq) | ~hours–days (feature crafting, niche-form choice, diagnostics) | ~2 min | < 0.1 s | see `summary.md` | see `summary.md` | `metrics.json` |
| **After: GBM (calibrated)** | ~1–2 days (pick model, tune, sanity-check calibration) | ~5–10 min | < 5 s | **see `summary.md`** | **see `summary.md`** | `metrics.json`, `suitability_map.png`, `before_after_bars.png` |

> Person-time figures are **illustrative estimates** of build effort, not measured
> benchmarks. The headline of the experiment is the AUC / Brier / suitability-
> correlation gain of the GBM over the GLM, reported exactly in
> `results/metrics.json` and reproduced by `run_before_after.py`.

---

## Honest caveats (the part an environmental statistician must state)

- **GLM is more interpretable.** A logistic-GLM coefficient table tells a peer
  reviewer exactly *how* temperature and precipitation enter the suitability
  function (sign, scale, quadratic curvature). A GBM tells you "feature
  importance" and a partial-dependence plot — useful, but it is not a closed-
  form niche expression.
- **GLM extrapolates more safely.** Outside the training climate envelope the
  GLM's quadratic surface degrades smoothly to "very unsuitable" everywhere; a
  GBM stays at whatever the nearest training leaf predicted, which can be
  arbitrary. **For climate-change projection** (training on present-day
  climate, predicting on novel future climate), this is a real risk for the
  GBM and the GLM may actually be safer.
- **Background sampling is load-bearing for both methods.** Presence-only data
  needs a defensible *background* / pseudo-absence sample over the study area
  (uniform from the accessible region, target-group background, or
  effort-corrected). Get it wrong and both the GLM and the GBM will fit the
  *sampling bias* rather than the niche. A GBM is particularly sensitive
  because it can chase fine background patterns.
- **AUC has known limitations for SDM** (Lobo et al., Jiménez-Valverde et al.).
  We report Brier (a proper scoring rule) and suitability correlation alongside
  AUC for that reason. None of the three is a single-number arbiter.
- **The win is *for the chosen niche regime*.** Our synthetic generator
  deliberately bakes an elevation-driven interaction the GLM cannot reach with
  only linear + squared terms. In datasets where the true niche is a clean
  separable quadratic, the GLM ties or wins — and is more interpretable.

This is exactly the `env-stats-validate` discipline: AI proposes the model;
the human chooses among GLM / GBM / NN with the niche structure, sampling
design, and downstream use-case (interpretation, projection, decision
support) in mind.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (smaller samples, fewer trees): finishes in a few seconds
python experiments/10_species_distribution/run_before_after.py --quick

# Committed full run (800 presences, 4000 background): a few seconds on CPU
python experiments/10_species_distribution/run_before_after.py
```

Useful flags: `--quick`, `--n-pres N`, `--n-bg M`, `--train-frac F`,
`--gbm-n-est K`, `--gbm-depth D`, `--gbm-lr LR`, `--seed S`.

Outputs land in `results/`:

- `metrics.json` — AUC, Brier, true-suitability correlation, prevalence
  calibration, GLM coefficients and GBM feature importances, plus the
  configuration and wall times for provenance.
- `suitability_map.png` — three-panel map: TRUE suitability vs the GLM
  predicted surface vs the GBM predicted surface.
- `before_after_bars.png` — AUC (higher is better) and Brier (lower is
  better) bars contrasting the two methods.
- `summary.md` — human-readable table.

### Tests (fast, CPU, < 30 s)

```bash
python -m pytest experiments/10_species_distribution/tests common/tests/test_sdm.py -q
```

The suite checks: generator shapes & determinism; both presence and background
classes present in the split; GLM and GBM both return finite AUC / Brier /
suitability-correlation; GBM AUC strictly greater than GLM AUC on the
held-out test set; GBM Brier no worse than GLM's; GBM suitability correlation
no worse than the GLM's.

---

## Swapping in real data

The pipeline consumes any presence + background table with the columns
`x`, `y`, `temperature`, `precipitation`, `elevation`, `y_label`. To use real
data:

1. **Presence**: pull species occurrences from **iNaturalist** (research-grade)
   or **GBIF** (`pygbif`) for the focal species and study area.
2. **Background / pseudo-absence**: draw uniformly from the accessible study
   area (or use a target-group background for sampling-bias correction).
3. **Covariates**: sample from gridded climate layers at every presence and
   background location — for example **ERA5** 2 m temperature and total
   precipitation (`cdsapi` + `xarray`), **WorldClim** bioclimatic variables,
   plus a **DEM** (SRTM / Copernicus DEM) for elevation.
4. Build a flat DataFrame with the six columns above and call
   `before.glm_logistic.run_before(train, test, grid_df)` and
   `after.gbm_sdm.run_after(train, test, grid_df)` exactly as the runner does.

Keep these column names and shapes and every downstream script (BEFORE, AFTER,
runner, tests) works unchanged. The generator's `REAL_DATA_NOTE` in
`common/synthetic_sdm.py` carries the same recipe inline.

---

## Method notes (generic, no invented citations)

Logistic regression with linear + quadratic terms is the textbook GLM SDM
baseline; gradient boosting for presence/background SDM (and isotonic
calibration of tree-ensemble probabilities) are standard, well-established
tools in the SDM / applied-statistics literature. The known limitations of
AUC as a sole SDM scoring rule are also well-documented (e.g. Lobo et al.;
Jiménez-Valverde et al.) — hence we report Brier and true-suitability
correlation alongside it. No specific arXiv identifiers, authors, dates or
benchmark statistics are asserted here — per the project's zero-hallucination
rule, only the verifiable mechanics and the experiment's own measured numbers
are claimed.
