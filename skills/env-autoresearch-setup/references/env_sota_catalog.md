# Environmental-statistics SOTA catalog (citation gate seed)

Backbone choices for the autoresearch loop must cite an anchor here. Only verifiable
papers/tools/datasets are listed. The five named papers carry arXiv IDs; foundation
models and platforms are named WITHOUT invented version numbers or dates. Anything not
independently confirmed is marked **(verify)** and must be checked against
`ledgers/CITATIONS-TO-VERIFY.md` before it appears on a slide.

## Verified anchor key

- ClimateLLM — arXiv:2502.11059
- CLLMate — arXiv:2409.19058
- AI co-scientist ("Towards an AI co-scientist") — arXiv:2502.18864
- AI-Scientist-v2 — arXiv:2504.08066
- EagleVision — arXiv:2503.23330
- Time-series foundation models — TimesFM, Chronos, MOMENT, Moirai
- Platforms — BioDT, DestinE
- Datasets — ERA5, GBIF, iNaturalist, Sentinel-2, CAMELS, OpenAQ

## Tier 1 — classical baselines (the "BEFORE" and the skill-score reference)

| Backbone | Task type | What it is | Anchor / note |
|---|---|---|---|
| Climatology / persistence | forecasting | Mean-of-history or last-value forecast; the reference for skill scores. | Standard NWP baseline |
| ARIMA / STL decomposition | forecasting | Trend+seasonality decomposition and classical forecasting. | `statsmodels` |
| Mann-Kendall + GEV/GPD | evt | Non-parametric trend test + extreme-value fit (block maxima / POT), return levels. | `scipy.stats`; EVT fit lib (verify) |
| Logistic regression | classification | Linear baseline for presence / change / regime. | `scikit-learn` |
| Random Forest on indices | classification | RF on hand-computed indices (NDVI/NDWI), the EO "BEFORE". | `scikit-learn`; see `common/synthetic_remote_sensing.compute_indices` |
| Kriging / Gaussian process | forecasting/spatial | Geostatistical interpolation with variograms. | `scikit-learn` GPs / GSTools (verify) |

## Tier 2 — modern / foundation models

| Backbone | Task type | What it is | Anchor / note |
|---|---|---|---|
| TimesFM | forecasting | Time-series foundation model; **zero-shot** forecasts, no per-series training. | TimesFM (verify exact version) |
| Chronos | forecasting | TS foundation model; zero-shot probabilistic forecasting. | Chronos (`chronos-forecasting`, verify version) |
| MOMENT | forecasting | TS foundation model for forecasting/representation. | MOMENT (verify) |
| Moirai | forecasting | Universal TS forecasting foundation model. | Moirai (verify) |
| Small CNN | classification | CNN on raw multispectral bands (texture), GPU "AFTER" for land cover. | runs on 4090; see Exp04 |
| LSTM / TCN | forecasting | Sequence models for fields/series and CAMELS-style rainfall-runoff. | CAMELS LSTM benchmark (verify) |
| EagleVision-style detector | classification (EO) | Remote-sensing object detection for EO imagery. | EagleVision (arXiv:2503.23330) |
| Structured-LLM extraction | classification (text) | LLM turns free text into species-interaction triples; scored vs a gold set. | project recipe; CLLMate (arXiv:2409.19058) context |

## Tier 3 — gradient-boosted trees (keep as SEPARATE backbones)

| Backbone | Task type | Note |
|---|---|---|
| XGBoost | classification/forecasting | Do not merge with the others |
| LightGBM | classification/forecasting | Separate backbone |
| CatBoost | classification/forecasting | Separate backbone |

## Paradigm anchors (for the loop itself, not a backbone)

- **AI co-scientist** (arXiv:2502.18864) and **AI-Scientist-v2** (arXiv:2504.08066)
  anchor the "agent runs the research loop with rigor gates" paradigm.
- **ClimateLLM** (arXiv:2502.11059) anchors LLM use for climate text/tasks.
- **DestinE / BioDT** anchor digital-twin-scale model output (Earth / biodiversity).

> Citation gate rule: an experiment that proposes a backbone with no anchor here (and
> no `(verify)`-tagged stand-in) does NOT pass the gate and must not run.
