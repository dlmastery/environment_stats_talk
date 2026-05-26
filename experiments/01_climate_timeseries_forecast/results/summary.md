# Experiment 01 — Climate time-series forecasting: BEFORE vs AFTER

- Device: `cuda`  |  synthetic years: 4  |  split: 1021/220/219 (train/val/test)
- AFTER model: LSTM (hidden=24, lookback=14, epochs≤3)
- Quick mode: True

## Metrics by horizon

| Horizon | Method | RMSE (°C) | MAE (°C) | ACC |
|--------:|--------|----------:|---------:|----:|
| 1d | before: persistence | 1.973 | 1.556 | 0.962 |
| 1d | before: seasonal_naive | 13.330 | 11.557 | -0.348 |
| 1d | **after: LSTM** | **2.811** | 2.207 | 0.920 |
| 1d | _skill (LSTM vs persistence)_ | -0.424 | | |

## Wall time (this machine)

| Horizon | BEFORE (s) | AFTER (s) |
|--------:|-----------:|----------:|
| 1d | 0.00 | 10.41 |

Skill score = 1 − RMSE(after) / RMSE(persistence); >0 means the LSTM beats the naive persistence reference. Artifacts: `forecast_plot.png`, `before_after_bars.png`, `metrics.json`.
