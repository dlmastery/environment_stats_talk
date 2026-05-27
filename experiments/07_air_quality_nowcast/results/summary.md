# Experiment 07 -- PM2.5 air-quality nowcast: BEFORE vs AFTER

- Synthetic station: 240 days @ freq=`h`  |  split: 4031/865/864 (train/val/test)
- Eval targets (test steps): 864  |  AFTER model: `gbm` (21 features)  |  quick: False
- Exceedance (spike) threshold = 90% percentile of observed test PM2.5 = 76.2 ug/m^3

## Nowcast skill (one-step-ahead, test period)

| Method | RMSE (ug/m^3) ↓ | MAE ↓ | Skill vs persistence ↑ | Spike F1 ↑ |
|--------|----------------:|------:|-----------------------:|-----------:|
| Before: persistence | 17.399 | 9.504 | +0.000 | 0.736 |
| Before: linear AR | 18.101 | 12.917 | -0.040 | 0.750 |
| Before: ARIMA | 17.399 | 11.878 | +0.000 | 0.750 |
| **After: GBM + weather** | **12.142** | **6.298** | **+0.302** | **0.847** |
| _Ablation: GBM history-only (no weather)_ | 13.635 | 8.070 | +0.216 | 0.777 |
| _AFTER − best-before (arima)_ | -5.257 | -5.580 | +0.302 | +0.097 |

Skill vs persistence = `1 - RMSE_model/RMSE_persistence` (1 = perfect, 0 = ties persistence, <0 = worse). Spike F1 = F1 for detecting threshold exceedances (episodes). The AFTER model reads the **weather covariates** (wind ventilation, temperature, boundary-layer height) + calendar features, so it leads the ventilation-driven spikes the history-only baselines lag. The history-only **ablation** row shows the win is driven by the covariates: strip them and the ML model falls back toward the classical baselines.

## Wall time (this machine)

- BEFORE: 32.03 s   |   AFTER: 9.20 s

Artifacts: `timeseries_plot.png`, `before_after_bars.png`, `metrics.json`.
