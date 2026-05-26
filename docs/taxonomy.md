# Taxonomy of Environmental Statistics (Environmetrics) — and Where AI / Claude Code Fits

**Audience:** environmental statisticians (TIES-type). **Purpose:** a slide-ready map of the field with a concrete AI/agentic-coding hook for every node, an anchor we can defend, a public dataset, a maturity tag, and the repo experiment that demonstrates it.

**How to read each row:** *what it is* · *where AI / Claude Code fits* · *verified tool/model anchor* · *public dataset* · *maturity* · *repo Exp*.

**Maturity legend:** `production-ready` = routinely deployable today · `emerging` = works, still maturing/validating · `opportunity` = credible but largely undemonstrated in environmetrics.

**Anchor discipline (scientists' audience):** only verifiable papers/tools/datasets are cited. arXiv IDs are given for the five named papers; foundation-model and platform names are given without invented version numbers or dates. Anything not independently confirmed is marked **(verify)** and must be checked against `ledgers/CITATIONS-TO-VERIFY.md` before it reaches a slide.

**Verified anchor key (used throughout):**
ClimateLLM = arXiv:2502.11059 · CLLMate = arXiv:2409.19058 · AI co-scientist = arXiv:2502.18864 · AI-Scientist-v2 = arXiv:2504.08066 · EagleVision = arXiv:2503.23330 · TS foundation models = TimesFM / Chronos / MOMENT / Moirai · platforms = BioDT, DestinE · datasets = ERA5, GBIF, iNaturalist, Sentinel-2, CAMELS, OpenAQ.

**Experiment key:** Exp01 climate forecasting · Exp02 extremes/trends · Exp03 biodiversity text · Exp04 remote sensing · Exp05 autoresearch loop.

---

## Branch 1 — Core statistical methods

| Sub-node | (a) What it is | (b) Where AI / Claude Code fits | (c) Tool/model anchor | (d) Dataset | (e) Maturity | (f) Exp |
|---|---|---|---|---|---|---|
| Spatial / geostatistics | Modelling spatially correlated fields (kriging, variograms, Gaussian processes). | Claude Code scaffolds the variogram-fit + kriging pipeline and benchmarks it against an ML interpolator; agent handles CV folds and leave-one-out diagnostics. | scikit-learn GPs / GSTools (verify) | ERA5 reanalysis grids; Sentinel-2 | emerging | Exp01, Exp04 |
| Time-series analysis | Trend/seasonality decomposition, autocorrelation, forecasting of environmental series. | Claude Code writes & runs the ARIMA/STL baseline, then drops in a TS **foundation model for a zero-shot forecast** with no per-series training. | TimesFM / Chronos / MOMENT / Moirai | ERA5; CAMELS streamflow | production-ready | Exp01 |
| Multivariate / high-dimensional | Joint structure across many variables (PCA/EOF, CCA, clustering, dimension reduction). | Agent automates EOF/PCA extraction, variance-explained tables, and clustering sweeps; flags collinearity and leakage before modelling. | scikit-learn / xarray-EOF (verify) | ERA5 multi-variable fields | production-ready | Exp01, Exp04 |
| Sampling design | Choosing where/when/how to sample for valid inference (stratified, spatially balanced, adaptive). | Claude Code generates candidate designs (GRTS-style spatially balanced), simulates estimator variance, and compares power under each. | spsurvey / GRTS (verify) | GBIF occurrence grids | opportunity | Exp03 |
| Extreme value theory | Tails and rare events: GEV/GPD fits, return levels, block maxima, POT. | Claude Code writes & runs the **Mann-Kendall + GEV/GPD pipeline**, computes return-level CIs, and runs the rigor checklist (stationarity, threshold choice). | scipy.stats / `lmoments`-style EVT fit (verify) | ERA5 temperature/precip extremes | emerging | Exp02 |
| Bayesian hierarchical | Multi-level models with partial pooling and explicit priors for uncertainty. | Agent drafts the PyMC/Stan model, runs MCMC, and checks convergence (R-hat, ESS); can compare against amortized/variational inference. | PyMC / Stan (verify) | CAMELS; OpenAQ station hierarchies | emerging | Exp02 |
| Uncertainty quantification | Calibrated predictive intervals, conformal prediction, ensemble spread. | Claude Code wraps any model in conformal prediction, produces coverage plots, and audits calibration vs nominal level. | MAPIE / conformal-prediction libs (verify) | ERA5; CAMELS | emerging | Exp01, Exp02 |

---

## Branch 2 — Application domains

| Sub-node | (a) What it is | (b) Where AI / Claude Code fits | (c) Tool/model anchor | (d) Dataset | (e) Maturity | (f) Exp |
|---|---|---|---|---|---|---|
| Climate & atmospheric | Statistics of weather/climate fields, forecasting, reanalysis analysis. | Foundation TS model gives a **zero-shot baseline**; Claude Code orchestrates training of LSTM/TCN and computes latitude-weighted RMSE/ACC. ClimateLLM frames LLM use for climate text/tasks. | ClimateLLM (arXiv:2502.11059); TimesFM/Chronos | ERA5 | production-ready | Exp01 |
| Water / hydrology | Streamflow, rainfall-runoff, water-balance statistics. | Agent builds the CAMELS-style LSTM vs conceptual-model comparison; handles basin grouping splits to avoid leakage. | LSTM rainfall-runoff (CAMELS benchmark) | CAMELS | production-ready | Exp01 (transfer) |
| Soil / geochemistry | Spatial soil properties, contamination mapping, compositional data. | Claude Code sets up kriging/GP interpolation and compositional-data transforms; benchmarks ML interpolators. | GSTools / `compositions` (verify) | Sentinel-2 derived indices; ISRIC SoilGrids (verify) | opportunity | Exp04 |
| Biodiversity & ecology | Species occurrence/abundance, SDMs, interaction networks. | Structured-LLM extraction turns literature text into species-interaction triples; Claude Code assembles the network and scores precision/recall vs a gold set. | LLM structured extraction (project recipe) | GBIF; iNaturalist | emerging | Exp03 |
| Pollution / air–water quality | Concentration fields, exceedance stats, source apportionment, nowcasting. | Agent builds PM2.5 nowcasting + exceedance-probability pipeline; automates station QA/QC and gap filling. | scikit-learn / statsmodels | OpenAQ | production-ready | Exp02 (extremes/exceedance) |
| Marine / coastal | Sea-surface fields, coastal change, ocean indicators. | Claude Code runs change detection on coastal imagery and SST trend tests; EagleVision-style remote-sensing detection for vessels/objects. | EagleVision (arXiv:2503.23330) | Sentinel-2 | emerging | Exp04 |
| Socio-ecological | Coupled human–environment systems, exposure, land-use/livelihood links. | Multimodal fusion (raster + text) links environmental state to socio-economic context; agent joins heterogeneous tables and documents assumptions. | CLLMate (arXiv:2409.19058) | GBIF + OpenAQ + Sentinel-2 joins | opportunity | Exp03, Exp04 |

---

## Branch 3 — Data ecosystems

| Sub-node | (a) What it is | (b) Where AI / Claude Code fits | (c) Tool/model anchor | (d) Dataset | (e) Maturity | (f) Exp |
|---|---|---|---|---|---|---|
| In-situ / sensors | Ground stations, gauges, monitoring networks (point time series). | Claude Code automates fetch → QA/QC → gap-fill → harmonize for station networks; flags drift and outliers statistically. | OpenAQ / station APIs | OpenAQ; CAMELS gauges | production-ready | Exp02 |
| Remote sensing / EO | Satellite raster imagery and derived products. | Agent builds the RF-on-indices vs CNN-on-4090 land-cover pipeline; computes change maps and embeddings. EagleVision anchors EO object detection. | EagleVision (arXiv:2503.23330) | Sentinel-2 | production-ready | Exp04 |
| Citizen science | Volunteer-contributed observations (presence-heavy, biased). | Claude Code applies detection-bias correction and effort weighting; structured-LLM cleans free-text records. | GBIF/iNaturalist pipelines | GBIF; iNaturalist | emerging | Exp03 |
| Simulation / model output | Reanalyses, ESM runs, digital-twin scenarios. | Agent post-processes large model output (xarray), computes diagnostics, and runs bias-correction vs observations. DestinE/BioDT anchor twin-scale model output. | DestinE; BioDT | ERA5 (reanalysis) | emerging | Exp01 |
| Multimodal fusion | Combining raster + text + tabular + time-series into one analysis. | CLLMate-style fusion of imagery + event text; Claude Code aligns modalities and builds event descriptions. | CLLMate (arXiv:2409.19058) | Sentinel-2 + event text | opportunity | Exp03, Exp04 |

---

## Branch 4 — Cross-cutting & policy

| Sub-node | (a) What it is | (b) Where AI / Claude Code fits | (c) Tool/model anchor | (d) Dataset | (e) Maturity | (f) Exp |
|---|---|---|---|---|---|---|
| Indicators / SEEA / SDGs | Standardized environmental-economic accounting and SDG indicators. | Agent computes indicator series from raw data and auto-drafts a reproducible indicator notebook + provenance log. | UN SEEA framework (verify); SDG indicator defs (verify) | OpenAQ; GBIF | opportunity | Exp02 (indicator trends) |
| Detection & attribution | Statistically separating signal (e.g., forced trend) from noise; attributing change. | Claude Code runs trend tests (Mann-Kendall, change-point) with multiple-testing control and documents the attribution logic. | scipy / `ruptures` change-point (verify) | ERA5 | emerging | Exp02 |
| Risk / vulnerability | Hazard × exposure × vulnerability; return periods feeding risk metrics. | Agent links EVT return levels to exposure layers and produces risk maps + uncertainty bands. | EVT fit + GIS overlay (verify) | ERA5; Sentinel-2 | opportunity | Exp02, Exp04 |
| Sustainability | Long-horizon resource/footprint trajectories and scenario assessment. | Claude Code builds scenario what-if comparisons on a (synthetic) ecosystem/digital-twin and tabulates trade-offs. | DestinE; BioDT | (synthetic + ERA5) | opportunity | Exp05 |
| Environmental justice | Distribution of environmental burdens/benefits across populations. | Agent joins pollution exposure (OpenAQ) to demographic strata, runs disparity statistics, and audits for confounding. | OpenAQ + census joins (verify) | OpenAQ | opportunity | — |

---

## Branch 5 — Emerging frontiers

| Sub-node | (a) What it is | (b) Where AI / Claude Code fits | (c) Tool/model anchor | (d) Dataset | (e) Maturity | (f) Exp |
|---|---|---|---|---|---|---|
| AI/ML-augmented environmetrics | Foundation models + ML as drop-in baselines/accelerators alongside classical stats. | Foundation TS model gives zero-shot forecasts; Claude Code runs the before/after head-to-head with classical baselines and keeps the human as final arbiter of assumptions. | TimesFM / Chronos / MOMENT / Moirai; ClimateLLM (arXiv:2502.11059) | ERA5; CAMELS | emerging | Exp01, Exp02 |
| Agentic / autonomous discovery | An agent runs the research loop (hypothesize → experiment → analyze) with hard rigor gates. | Claude Code **is the researcher**: a Diagnose→Cite→Hypothesize→Predict→Execute→Analyze→Checkpoint loop with citation + reasoning gates; AI co-scientist and AI-Scientist-v2 anchor the paradigm. | AI co-scientist (arXiv:2502.18864); AI-Scientist-v2 (arXiv:2504.08066) | ERA5 + synthetic | emerging | **Exp05** |
| Digital twins | Continuously updated virtual replicas of an environmental system for scenario testing. | Agent drives scenario sweeps and uncertainty propagation on a twin; reports trade-offs. DestinE (Earth) and BioDT (biodiversity) anchor real twin platforms. | DestinE; BioDT | DestinE/BioDT outputs (verify) | emerging | Exp05 |
| Explainable / sustainable AI | Interpretable model behaviour + accounting for compute/energy cost of the methods. | Claude Code adds SHAP/feature-attribution and conformal calibration, and logs runtime/energy so "100× faster" includes a compute-cost footnote. | SHAP / conformal libs (verify) | (any above) | opportunity | Exp04, Exp05 |

---

## Top high-leverage nodes (shortlist)

Where the AI/agentic-coding payoff is largest **and** the anchor is most defensible for a scientists' audience:

1. **Time-series analysis → foundation-model zero-shot forecasting (Branch 1 / Exp01).** Biggest "before/after" gap: classical ARIMA/STL vs a TS foundation model (TimesFM/Chronos/MOMENT/Moirai) with no per-series training. Production-ready, strong anchors.
2. **Extreme value theory + detection & attribution → agent-run Mann-Kendall + GEV/GPD pipeline (Branch 1 & 4 / Exp02).** High statistical rigor, directly relevant to climate-risk policy, and a clean demonstration of Claude Code writing/running a defensible pipeline with multiple-testing control.
3. **Agentic / autonomous discovery → the autoresearch loop (Branch 5 / Exp05, flagship).** The headline "science 100× faster" claim, anchored by AI co-scientist (2502.18864) and AI-Scientist-v2 (2504.08066), with hard citation/reasoning gates preserving rigor.
4. **Remote sensing / EO → RF-vs-CNN land cover + EagleVision (Branch 2 & 3 / Exp04).** Production-ready, runs on the 4090, EagleVision (2503.23330) anchors EO detection; tangible change maps for slides.
5. **Multimodal fusion → CLLMate-style raster+text event description (Branch 2 & 3 / Exp03).** Newest capability with a concrete paper anchor (CLLMate 2409.19058); turns unstructured biodiversity/event text into structured, analyzable outputs.

---

*Note: rows tagged **(verify)** for tooling/datasets indicate the concept is sound but the exact library/version/dataset must be confirmed against `ledgers/ITEMS.md` and `ledgers/CITATIONS-TO-VERIFY.md` before appearing on a slide. No dates, version numbers, or quantitative claims are asserted here beyond the five verified arXiv IDs.*
