# Experiment 01 — Climate time-series forecasting: BEFORE vs AFTER

- Device: `cuda`  |  synthetic years: 20  |  split: 5110/1095/1095 (train/val/test)
- AFTER model: LSTM (hidden=48, lookback=30, epochs≤80)
- Quick mode: False

## Metrics by horizon

| Horizon | Method | RMSE (°C) | MAE (°C) | ACC |
|--------:|--------|----------:|---------:|----:|
| 1d | before: persistence | 1.876 | 1.433 | 0.963 |
| 1d | before: seasonal_naive | 3.713 | 3.010 | 0.853 |
| 1d | before: sarima | 1.826 | 1.418 | 0.964 |
| 1d | **after: LSTM** | **1.767** | 1.378 | 0.967 |
| 1d | _skill (LSTM vs persistence)_ | 0.058 | | |
| 7d | before: persistence | 3.580 | 2.847 | 0.865 |
| 7d | before: seasonal_naive | 3.713 | 3.010 | 0.853 |
| 7d | before: sarima | 2.054 | 1.636 | 0.955 |
| 7d | **after: LSTM** | **2.811** | 2.241 | 0.914 |
| 7d | _skill (LSTM vs persistence)_ | 0.215 | | |
| 14d | before: persistence | 4.308 | 3.415 | 0.806 |
| 14d | before: seasonal_naive | 3.713 | 3.010 | 0.853 |
| 14d | before: sarima | 2.096 | 1.667 | 0.953 |
| 14d | **after: LSTM** | **2.937** | 2.335 | 0.905 |
| 14d | _skill (LSTM vs persistence)_ | 0.318 | | |

## Wall time (this machine)

| Horizon | BEFORE (s) | AFTER (s) |
|--------:|-----------:|----------:|
| 1d | 32.62 | 39.20 |
| 7d | 31.98 | 21.97 |
| 14d | 27.03 | 3.12 |

Skill score = 1 − RMSE(after) / RMSE(persistence); >0 means the LSTM beats the naive persistence reference. Artifacts: `forecast_plot.png`, `before_after_bars.png`, `metrics.json`.
