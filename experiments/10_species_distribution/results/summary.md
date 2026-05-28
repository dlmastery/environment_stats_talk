# Experiment 10 - Species distribution modelling: BEFORE vs AFTER

- Records: presences=800, background=4000, train=3360, test=1440
- Prevalence: train=0.171, test=0.158
- Covariates: temperature, precipitation, elevation
- Grid: 60 x 60 = 3600 cells
- Quick mode: False, seed: 0

## Headline metrics

| Method | AUC (higher better) | Brier (lower better) | True-suitability corr | Prevalence calibration gap |
|--------|--------------------:|---------------------:|-----------------------:|---------------------------:|
| Before: GLM (logistic + sq) | 0.7050 | 0.1243 | 0.6964 | 0.0131 |
| **After: GBM** | **0.7277** | **0.1224** | **0.8839** | 0.0140 |

AFTER - BEFORE: AUC +0.0226, Brier -0.0019 (negative = better), suitability corr +0.1876.

Wall time: before 0.033s, after 8.222s.

_AUC: higher is better (1 = perfect ranking, 0.5 = random). Brier: lower is better (0 = perfect probability). suitability_corr: Pearson correlation between predicted P(presence) and the TRUE noise-free suitability on the evaluation grid._

Artifacts: `metrics.json`, `suitability_map.png`, `before_after_bars.png`.
