# Exp01 foundation-model zero-shot results

Every number was produced by `python experiments/01_climate_timeseries_forecast/run_all_foundation.py` on this machine (Windows, RTX 4090 Laptop). Same synthetic 20-year daily temperature series and same chronological 70/15/15 split as the other Exp01 baselines.

| Model | h=1 RMSE | h=7 RMSE | h=14 RMSE | h=1 skill | h=7 skill | h=14 skill | wall-time (s) | device |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Chronos-Bolt-small (zero-shot) | 1.823 | 3.420 | 4.248 | 0.028 | 0.026 | -0.004 | 2.64 | — |
| TimesFM-2.0-500m (zero-shot) | 1.817 | 3.204 | 3.831 | 0.031 | 0.087 | 0.094 | 9.10 | cuda |
| MOMENT-1-small (reconstruction zero-shot) | 8.754 | 8.754 | 8.740 | -3.668 | -1.494 | -1.066 | 1.13 | cuda |
| Moirai-1.0-R-small (zero-shot) | 6.766 | 9.114 | 6.178 | -2.608 | -1.597 | -0.460 | 6.58 | cuda |
| ClimateLLM (no public release) | — | — | — | — | — | — | — | no public release |

## Skipped
- **climatellm**: ClimateLLM (arXiv:2502.11059) has no public code or pretrained weights release. Paper only references google-research/weatherbench2 as the data source; no model GitHub or HF repo is published. Skipping.

## How to read this

- *skill_vs_persistence* = 1 - RMSE_model / RMSE_persistence (positive = beats persistence).
- This is **zero-shot**: no fine-tuning on the temperature series.
- On a strongly-seasonal synthetic series, a well-fit SARIMA is hard to beat; Chronos / TimesFM / MOMENT / Moirai were not exposed to the seasonal structure at training time. The point is to run the comparison honestly, fast.