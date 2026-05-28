# Experiments Index — All 9 Before/After Experiments

A one-stop index of every experiment in this repository. Each is a self-contained
before/after pair: a **BEFORE** (the traditional environmental-statistician's
approach) and an **AFTER** (Claude Code + an AI-for-science model writing, running,
and benchmarking the workflow), with committed `results/` (a `metrics.json`, plots,
and a `summary.md`) reproducible from synthetic data with deterministic seeds.

> **Run every command from the repo root** (the folder containing `common/` and
> `experiments/`) so `import common` resolves. Every headline number below is
> committed in that experiment's `results/` and summarized in `RESULTS.md` — none
> is invented.
>
> **What "AFTER wins" means.** The verdicts below are **mixed on purpose.** AFTER's
> universal win is the collapse in *human effort* (days of bespoke coding → one
> command in minutes) plus the *breadth of rigor* added. Whether the AFTER *model*
> beats the classical one is reported honestly, case by case — sometimes it does
> not, and that is the point.

The experiments are grouped by the **taxonomy branch** they demonstrate (see
`docs/taxonomy.md`).

---

## At-a-glance scoreboard

| # | Task | BEFORE → AFTER headline | GPU? | Honest verdict |
|---|------|--------------------------|:----:|----------------|
| 01 | Daily temperature forecasting | persistence RMSE 4.31 → LSTM 2.94 (skill +0.318) at 14-d; **SARIMA 2.10** | auto-CUDA (optional) | **SARIMA wins at ≥7-day lead** — classical beats the net; the win is the *fair, fast comparison* |
| 02 | Precipitation extremes & trend | empirical 100-yr 48.6 mm → GEV **55.8 mm [46.7–80.2]**; MK p=0.0011 | no | **Rigor win** — honest uncertainty; empirical method was biased low |
| 03 | Species-interaction text extraction | keyword R **0.10** → structured **1.00** (synthetic ceiling) | no | **Recall 10% → 100%** on a closed corpus (a ceiling, not a real-world number) |
| 04 | Land cover + change detection | easy: RF = CNN = 1.00; **hard mode: RF 0.642 → CNN 0.998 (+0.356 acc)** | auto-CUDA (hard mode real-4090) | **Two regimes, both honest** — tie on easy classes, real CNN win on texture-only classes |
| 05 | **Autoresearch loop (flagship)** | gated 6-experiment loop in ~38 s; champion **+8.0% skill** (1.95→1.79 °C) | CPU (4090 to scale) | **Process win** — gates block a shallow guess; the agent rejected a wrong "obvious" idea |
| 06 | Spatial interpolation | IDW 4.94 → kriging 2.91 → **RF + covariate 2.20** | no | **AFTER win with a caveat** — −0.70 RMSE (~24%), but kriging keeps a variance surface |
| 07 | PM2.5 air-quality nowcast | persistence/ARIMA 17.4 → **GBM + weather 12.1** | no | **Clean AFTER win** — −5.3 RMSE, +0.30 skill, spike-F1 0.74→0.85 |
| 08 | Rainfall-runoff streamflow | linear/bucket NSE 0.14 → **LSTM NSE 0.70** | auto-CUDA (optional) | **Clean AFTER win** — +0.56 NSE / +0.51 KGE (the classic LSTM-hydrology result) |
| 09 | Bayesian hierarchical (MCMC vs amortized) | MH **11.6 s/dataset** → amortized **0.008 s** (train-once 24.5 s) | no | **~1451× faster at scoring**, coverage parity, wider intervals — honest speed/sharpness tradeoff |
| 10 | Species distribution modeling | GLM AUC **0.71** → GBM **0.73**; suit-corr **0.70 → 0.88** | no | **AFTER win on all 3 metrics** (biggest gain on the niche-interaction surface); GLM still safer for extrapolation |
| 12 | Conformal prediction intervals | normal-theory gap 0.033 → **conformal 0.004 (~7×)** | no (CPU only) | **Calibration + sharpness win** — distribution-free coverage, narrower bands |

---

## Taxonomy Branch 1 — Core statistical methods

### Exp 02 — Climate extremes & trend detection

- **One line:** Detect trends and characterize extremes (annual maxima, ETCCDI
  indices Rx5day/R95p, Mann-Kendall + Sen's slope, 20/50/100-yr return levels) in a
  40-year synthetic daily precipitation record whose tail intensifies over time.
- **BEFORE:** the manual EDA day — group-by-year maxima, a fixed 95th-percentile
  threshold, a from-scratch Mann-Kendall + Sen's slope, and a return level read off
  the **empirical** quantile (point estimates only, no fit, no CIs, cannot
  extrapolate past the record).
- **AFTER:** a block-maxima **GEV fit (MLE)** with **bootstrap confidence
  intervals**, Mann-Kendall trend tests on all three indices, and a human-in-the-loop
  `validate()` gate that flags stationarity/autocorrelation/n-blocks/shape/multiple-
  testing issues.
- **Headline:** empirical 100-yr return level **48.6 mm** (biased low, saturates near
  the largest observed maximum) vs **GEV 100-yr 55.8 mm with a [46.7–80.2] bootstrap
  CI**; Mann-Kendall trend **p = 0.0011**. The GEV *extrapolates* and the CI honestly
  *widens* with return period.
- **Run:** `python experiments/02_extreme_value_trends/run_before_after.py`
  (`--quick` for a fast smoke run).
- **Key artifacts:** `results/return_levels.png` (GEV curve + 95% bootstrap CI band),
  `results/trend_plot.png`, `results/metrics.json`, `results/summary.md`.
- **GPU?** No — CPU only.
- **Honest verdict:** **Rigor win.** Same question, far more honest uncertainty; the
  empirical method was biased low, and the validation gate flags its own stationarity
  violation. AFTER is faster *and* more rigorous, not looser.

### Exp 06 — Spatial interpolation (geostatistics)

- **One line:** Map an environmental field over a 2-D domain from sparse scattered
  station observations, scored against a dense held-out grid of the true field.
- **BEFORE:** coordinate-only interpolators — **Inverse-Distance Weighting** and
  **ordinary kriging** (a Gaussian process with an RBF covariance + nugget, the
  modern automatic stand-in for variogram fitting), plus the classic Matheron
  empirical variogram. Kriging also yields a calibrated **predictive-variance
  surface**.
- **AFTER:** a **covariate-aware** RandomForest (GBM/MLP selectable) given an
  auxiliary **elevation** covariate at every prediction cell plus engineered
  interaction features — capturing the short-scale lapse response that a coords-only
  smoother must average across.
- **Headline (180 stations, 45×45 grid, RF-400, seed 0):** IDW **4.94** → ordinary
  kriging **2.91** → **RF + covariate 2.20** RMSE. AFTER − kriging = **−0.70 RMSE
  (~24%)**.
- **Run:** `python experiments/06_spatial_interpolation/run_before_after.py`
  (`--model gbm` for a larger margin; `--quick` for a fast run).
- **Key artifacts:** `results/predicted_surface.png` (covariate / true / kriging / ML
  maps), `results/variogram.png`, `results/before_after_bars.png`,
  `results/metrics.json`, `results/summary.md`.
- **GPU?** No — CPU only, finishes in seconds.
- **Honest verdict:** **AFTER win with a caveat.** A real −0.70 RMSE over a *real*
  kriging baseline (not a strawman), **but** kriging keeps a calibrated variance
  surface the tree model lacks. The true hero is **regression-kriging** (the
  covariate folded *into* the geostatistical model) — lower error *and* a calibrated
  uncertainty surface.

### Exp 12 — Conformal prediction intervals (uncertainty quantification)

- **One line:** Put a prediction interval around one transparent daily-temperature
  point forecaster two ways, then **check empirical coverage** against the nominal
  level under heavy-tailed, heteroscedastic residuals.
- **BEFORE:** textbook Gaussian / normal-theory bands `ŷ ± z·σ̂` (correct only if
  residuals are i.i.d. Normal with constant variance — which they are not here).
- **AFTER:** **split (inductive) conformal** intervals from the empirical quantile of
  held-out calibration residuals, plus a **locally-adaptive (normalized) conformal**
  variant whose width breathes with the seasonal residual spread. Distribution-free,
  finite-sample marginal coverage.
- **Headline (20 yr, seed 0):** mean absolute calibration gap **normal-theory 0.0329
  → split conformal 0.0044 (~7× tighter)**, *and* narrower bands (80% width
  **6.37 → 5.22 °C**). At 80% the normal band over-covers (0.869 vs 0.80); at 95% the
  two agree.
- **Run:** `python experiments/12_conformal_uncertainty/run_before_after.py`
  (`--quick` tells the same story with a slightly larger gap).
- **Key artifacts:** `results/coverage_plot.png` (distance from `y = x` = miscalib),
  `results/interval_plot.png` (conformal band visibly tighter, normalized band
  breathes), `results/metrics.json`, `results/summary.md`.
- **GPU?** No — CPU only, no torch.
- **Honest verdict:** **Calibration + sharpness win.** Distribution-free finite-sample
  marginal coverage with narrower intervals. Caveats stated honestly: the guarantee is
  *marginal* not conditional, residuals are weakly serially dependent (breaking strict
  exchangeability), and very high levels need enough calibration points.

---

## Taxonomy Branch 1 / Branch 2 — Time-series forecasting (climate & hydrology)

### Exp 01 — Climate time-series forecasting

- **One line:** Forecast daily 2 m air temperature `h` days ahead on synthetic
  ERA5-like data; build and skill-score four forecasters in one run.
- **BEFORE:** **persistence** (random walk), **seasonal-naive**, and **(S)ARIMA**
  (SARIMAX, with an `AutoReg` fallback) — the classical toolkit.
- **AFTER:** a compact **1-layer LSTM** (~48 hidden) over a 30-day lookback +
  sin/cos day-of-year, trained with AdamW + early stopping; plus an optional
  zero-shot **foundation-model** baseline (Chronos / TimesFM) that is skipped
  cleanly when absent.
- **Headline (20 yr, seed 0, device=cuda):** at the 14-day lead, persistence RMSE
  **4.31** → LSTM **2.94** (skill **+0.318**), but **SARIMA 2.10** wins; at 7-day
  lead SARIMA **2.05** also beats the LSTM **2.81**; at 1-day all three are within a
  hair (persistence 1.876, SARIMA 1.826, LSTM 1.767).
- **Run:** `python experiments/01_climate_timeseries_forecast/run_before_after.py`
  (`--quick` for the 5-minute first run; `--epochs 80` for the GPU headline pass).
- **Key artifacts:** `results/metrics.json`, `results/forecast_plot.png`,
  `results/before_after_bars.png`, `results/summary.md`.
- **GPU?** Optional — the LSTM auto-uses CUDA (committed run `device=cuda`) but runs
  fine on CPU.
- **Honest verdict:** **SARIMA wins at ≥ 7-day lead** — classical statistics beats the
  net on this near-AR(1) single-station problem. The win is the *fair, fast,
  reproducible comparison* of all four methods in one command. (Deep dive:
  `docs/TUTORIAL_run_an_experiment.md`.)

### Exp 08 — Rainfall-runoff streamflow forecasting

- **One line:** Predict daily catchment streamflow from meteorological forcing
  (precipitation, temperature, PET) on synthetic but physically-plausible data — the
  classic case where ML genuinely beats classical hydrology.
- **BEFORE:** **linear regression** on lagged precipitation + an Antecedent
  Precipitation Index + same-day temperature/PET, and a **conceptual single
  linear-reservoir bucket** (recession constant grid-searched on train). A fixed
  linear map that structurally underfits the nonlinear storage memory.
- **AFTER:** a compact **LSTM rainfall-runoff model** (1 layer, ~32 hidden) over a
  lookback window of `[precip, temp, pet]` + sin/cos day-of-year, with AdamW +
  gradient clipping + early stopping; predictions clipped at zero. It carries an
  internal **cell state** — the "how full is the catchment right now" memory the
  linear model cannot represent.
- **Headline:** linear/conceptual NSE **0.14** → **LSTM NSE 0.70**, i.e. **+0.56 NSE
  / +0.51 KGE**.
- **Run:** `python experiments/08_hydrology_streamflow/run_before_after.py`
  (`--quick` for a smoke run; `--epochs 150` for the GPU pass).
- **Key artifacts:** `results/metrics.json`, `results/hydrograph.png`,
  `results/before_after_bars.png`, `results/summary.md`.
- **GPU?** Optional — auto-CUDA, CPU fallback. The real LSTM-hydrology win comes from
  multi-basin (CAMELS) training, which is the natural 4090 scale-up.
- **Honest verdict:** **Clean AFTER win** — the well-established LSTM-hydrology result.
  The gap widens with more data; the LSTM carries catchment state (soil moisture,
  snow, routing) the linear model cannot.

---

## Taxonomy Branch 2 — Application domains (pollution, biodiversity, remote sensing)

### Exp 07 — PM2.5 air-quality nowcast

- **One line:** Nowcast hourly urban PM2.5 — the classical history-only time-series
  toolkit vs a model that reads the **weather covariates** that actually drive
  particulate pollution.
- **BEFORE:** **persistence**, **linear AR(p)**, and **ARIMA** — all using only the
  PM2.5 history, so all blind to the cause of pollution spikes.
- **AFTER:** a **GradientBoostingRegressor** (MLP selectable) over PM2.5 lags +
  contemporaneous & lagged **weather** (wind/ventilation, temperature, boundary-layer
  height/inversions) + rolling ventilation summaries + cyclical calendar features.
  Leak-free (no weather feature includes the target).
- **Headline (240 days hourly, seed 0):** best-before (ARIMA) RMSE **17.40** →
  **GBM + weather 12.14** (skill vs persistence **+0.302**), exceedance **spike-F1
  0.750 → 0.847 (+0.097)**. Ablation: the *same* GBM with weather removed lands at
  13.63 — proving the covariates, not the model class, drive the win.
- **Run:** `python experiments/07_air_quality_nowcast/run_before_after.py`
  (`--quick`, `--model mlp`, `--no-arima`, `--freq D` available).
- **Key artifacts:** `results/metrics.json`, `results/timeseries_plot.png`,
  `results/before_after_bars.png`, `results/summary.md`.
- **GPU?** No — scikit-learn on CPU, no GPU dependency at all.
- **Honest verdict:** **Clean AFTER win** (−5.3 RMSE, +0.30 skill, +0.097 spike-F1) —
  but the lesson is "**bring in the physical drivers**," not "ML beats statistics." An
  agentic workflow makes pulling/aligning/engineering meteorological covariates a
  minutes-long task instead of a multi-day one.

### Exp 03 — Biodiversity species-interaction text extraction

- **One line:** Read short citizen-science comments (iNaturalist-style field notes)
  and extract species interactions as typed triples `(subject, relation, object)`
  over a fixed relation vocabulary (`pollinates | feeds_on | parasitizes |
  competes_with | depends_on`).
- **BEFORE:** a **keyword / regex** matcher over a hand-listed set of active-voice
  verbs paired with the nearest species — brittle, misses passive voice and synonyms.
- **AFTER:** a **schema-constrained structured extractor** that normalizes
  verbs/synonyms, resolves passive voice to `agent → patient`, and binds to the typed
  vocabulary; assembles an interaction network. (An optional real-LLM variant uses the
  Anthropic SDK with tool use, gated behind `ANTHROPIC_API_KEY`; the deterministic
  offline `extract()` is the default and powers the committed numbers.)
- **Headline (n = 400, seed 0):** precision **0.667 → 1.000**, recall **0.104 →
  1.000**, F1 **0.181 → 1.000**. The recall gap **+0.90** is the finding.
- **Run:** `python experiments/03_biodiversity_text_extraction/run_before_after.py`
  (`--quick` for n = 60; `--n 800` for a larger corpus).
- **Key artifacts:** `results/metrics.json` (triple-level P/R/F1 + graph + triples),
  `results/before_after_bars.png`, `results/interaction_network.png`,
  `results/summary.md`.
- **GPU?** No — CPU only, offline, no API key for the committed run.
- **Honest verdict:** **Recall 10% → 100%** — but **1.0 is the ceiling on a closed
  synthetic corpus**, *not* a real-world number. On real data, expect lower and
  noisier results, which is exactly why the mandatory human-verification step (machine
  drafts, ecologist validates) is part of the workflow.

### Exp 04 — Remote-sensing land cover & change detection

- **One line:** Classify 5-band multispectral patches into five land-cover classes
  (`water, forest, cropland, urban, bare`) and run a deforestation change-detection
  mini-task — with a **hard mode** that makes the task texture-separable.
- **BEFORE:** the GIS-analyst **RandomForest-on-indices** pipeline — collapse each
  patch to four physically-motivated scalars (NDVI, NDWI, mean SWIR, brightness) and
  classify those. Transparent, strong on well-separated classes, but throws away all
  spatial **texture**.
- **AFTER:** a compact 3-block **CNN on the raw `(5, H, W)` reflectance cube** (AdamW +
  early stopping, auto-CUDA), capturing the texture the indices discard; plus an
  optional pretrained-encoder-embeddings + kNN route (returns a documented skip when no
  encoder is supplied), and an NDVI-difference **change detector**.
- **Headline:** on the **easy** default classes, RF = CNN = **1.00** (a tie — the value
  is effort/capability, not accuracy). On **hard mode** (two class pairs share a
  near-identical mean spectrum and differ only in spatial texture, RTX 4090 run):
  RF-on-indices **0.642** acc vs CNN-on-bands **0.998** acc — **+0.356 accuracy /
  +0.357 macro-F1**.
- **Run (default):**
  `python experiments/04_remote_sensing_landcover/run_before_after.py --quick --cpu`
  ; **hard mode:** `python experiments/04_remote_sensing_landcover/after/cnn_hard.py`
  (`--quick --cpu` for the fast version).
- **Key artifacts (default):** `results/confusion_matrix.png`, `results/change_map.png`,
  `results/before_after_bars.png`, `results/metrics.json`, `results/summary.md`;
  **(hard mode)** `results/hard/metrics.json`, `results/hard/summary.md`,
  `results/hard/confusion_matrix_hard.png`, `results/hard/before_after_hard_bars.png`.
- **GPU?** Optional auto-CUDA; the committed **hard-mode** numbers are from a real
  4090 run (device=cuda, 53,741-param CNN).
- **Honest verdict:** **Two regimes, both honest.** On separable classes the cheap
  baseline is already excellent (tie) — so no fabricated accuracy win is claimed; the
  before/after there is *analyst-hours of index engineering → minutes of agent code*.
  On texture-only classes the CNN delivers a **real +0.356 accuracy win** because RF
  cannot resolve the same-spectrum texture pairs.

---

## Taxonomy Branch 5 — Emerging frontiers (agentic / autonomous discovery)

### Exp 05 — AutoResearch Climate Loop (the FLAGSHIP)

- **One line:** Claude Code **is the researcher** — instead of a statistician
  hand-running a forecasting sweep over days, a gated, literature-grounded
  autoresearch loop drives itself through a sequence of experiments to improve a
  climate forecaster and reports the honest result.
- **BEFORE:** a statistician hand-runs ~6 experiments over days — scattered notebooks,
  "which split/metric/seed gave that number?", rigor that erodes under deadline,
  easy to fool yourself, a crash loses the mental thread.
- **AFTER:** a **gated 6-experiment loop** (Diagnose → Cite → Hypothesize → Predict →
  Execute → Analyze → Checkpoint) with hard gates: **Citation-Rigor** (a real
  reference with a 4-digit year, venue/arXiv-ID/title, and a relevance clause) and
  **Reasoning-Completeness** (word-count floors, a *mechanistic* hypothesis, a
  *numeric-range* prediction, a KEEP/DISCARD verdict), plus a **frozen composite
  fingerprint** (`942df6a8450d8d57`) that makes silently redefining the metric *raise*
  rather than improve the score (Goodhart guard), plus a crash-recovery checkpoint
  rewritten after every run.
- **Headline:** the entire loop runs in **~38 s of compute**; champion is **Exp5**, an
  **8.0% RMSE improvement over persistence** (RMSE **1.95 → 1.79 °C**, skill +0.080),
  with a **non-decreasing** champion composite the whole way; Exp6 (super-fold-by-
  regime robustness) correctly **DISCARDs** and leaves the champion intact. The loop
  also rejected a wrong "obvious" idea — *more* lags hurt this near-AR(1) series, so
  Exp3 went the *other* way (3 lags) on a parsimony argument.
- **Run:** `python experiments/05_autoresearch_climate/run_loop.py`
  (idempotent — wipes `results/` first).
- **Key artifacts:** `results/experiment_log.jsonl` (one JSON row per experiment),
  `results/reasoning_annotations.json` (the six full gate-passing reasoning blobs),
  `results/best_config.json` (the champion + frozen fingerprint),
  `results/project_autoresearch_checkpoint.md`, `champion_progress.png`,
  `research_journal.md`.
- **GPU?** CPU for the committed smoke run (tiny numpy MLP, runs offline in seconds);
  the **identical protocol** scales to the 4090 by swapping in a torch LSTM or a
  zero-shot TS foundation model (TimesFM / Chronos) and real ERA5 data — only the
  backbone and data change, never the gates/composite/splits/checkpoint.
- **Honest verdict:** **Process win.** The gates *enforce* a rigor a hurried human
  skips — the shallow-entry test proves an experiment with no citation/mechanism/
  numeric prediction never runs ("you cannot guess"). The headline is a *modest*
  +8.0% (reported plainly as modest, because one-step daily temperature is a
  near-persistence problem) — the value is the disciplined, leakage-aware,
  Goodhart-proof, crash-resilient *process*, not the size of the delta.
- **Credit:** methodology adapted from the user's own `dlmastery/autoresearch`
  (`generalized_ml_autoresearch`); runnable package `autoresearch_env/`; full protocol
  in `docs/autoresearch_protocol.md`.

---

## Taxonomy Branch 1 — Core statistical methods (continued)

### Exp 09 — Bayesian hierarchical: MCMC vs amortized inference
- **What:** partial-pooling of S stations' offsets in a Normal-Normal hierarchy. BEFORE: closed-form conjugate update + from-scratch Metropolis-Hastings posterior. AFTER: a small MLP trained on simulated datasets to output a per-station posterior (mean, log-std) instantly at scoring time.
- **Headline (full run, S=30):** MH **11.6 s/dataset**, coverage 1.00, RMSE-to-truth 0.15; amortized **0.008 s** at scoring (train-once 24.5 s) — **~1451× speedup at scoring**, coverage parity, intervals wider (RMSE 0.27).
- **Honest tradeoff:** classical is interpretable and correct for *any* dataset; amortized pays for the speed in conservatism and must be retrained if the model class changes — the one-time training cost is paid back from the 2nd dataset onward.
- **Run:** `python experiments/09_bayesian_vs_amortized/run_before_after.py --quick` (CPU, no keys).
- **Artifacts:** `results/{metrics.json, posterior_intervals.png, before_after_bars.png, summary.md}`.

## Taxonomy Branch 2 — Application domains (continued)

### Exp 10 — Species distribution modeling (SDM)
- **What:** presence/background SDM with three climate covariates (T, P, elevation) on a 2D grid; the true niche is a Gaussian in (T, P) with an elevation interaction the GLM cannot encode. BEFORE: logistic GLM with linear + squared terms (classical SDM). AFTER: calibrated GradientBoosting on the same covariates.
- **Headline:** AUC 0.705 → **0.728** (+0.023); Brier 0.124 → **0.122**; true-suitability correlation 0.696 → **0.884** (**+0.188** — the GBM recovers the interaction).
- **Honest caveats:** GLM is more interpretable and extrapolates more safely outside the training climate; GBM is sensitive to the background-sample choice and can over-extrapolate. AUC alone hides niche-shape quality — keep the suitability-correlation metric in view.
- **Run:** `python experiments/10_species_distribution/run_before_after.py --quick` (CPU, no keys). Real-data swap: iNaturalist/GBIF presences + ERA5/SRTM covariates.
- **Artifacts:** `results/{metrics.json, suitability_map.png, before_after_bars.png, summary.md}`.

---

## Reproduce everything

```bash
# all unit tests (each component in its own subprocess)
python run_all_tests.py

# then each experiment's committed results, e.g.
python experiments/01_climate_timeseries_forecast/run_before_after.py
python experiments/02_extreme_value_trends/run_before_after.py
python experiments/03_biodiversity_text_extraction/run_before_after.py
python experiments/04_remote_sensing_landcover/run_before_after.py
python experiments/05_autoresearch_climate/run_loop.py
python experiments/06_spatial_interpolation/run_before_after.py
python experiments/07_air_quality_nowcast/run_before_after.py
python experiments/08_hydrology_streamflow/run_before_after.py
python experiments/12_conformal_uncertainty/run_before_after.py
```

See `RESULTS.md` for the full committed scoreboard and the labor-vs-wall-clock
"speedup" table, `docs/taxonomy.md` for the field map, and each
`experiments/NN_*/README.md` for the deep dive and real-data swap.
