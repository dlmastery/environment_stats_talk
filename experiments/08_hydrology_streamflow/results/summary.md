# Experiment 08 — Rainfall-runoff streamflow forecasting: BEFORE vs AFTER

- Device: `cuda`  |  synthetic years: 25  |  split: 6387/1369/1369 (train/val/test)
- Eval targets (test days): 1369
- AFTER model: LSTM (hidden=32, lookback=45d, epochs≤60, ran 19)
- Quick mode: False

## Streamflow skill (test period)

| Method | NSE ↑ | KGE ↑ | RMSE (mm/day) ↓ |
|--------|------:|------:|----------------:|
| Before: linear (lagged precip) | 0.142 | 0.175 | 1.525 |
| Before: conceptual bucket | -0.114 | 0.175 | 1.738 |
| **After: LSTM** | **0.704** | **0.684** | **0.895** |
| _AFTER − BEFORE (linear)_ | +0.562 | +0.508 | -0.630 |

NSE = Nash-Sutcliffe Efficiency, KGE = Kling-Gupta Efficiency (both 1 = perfect; higher is better). RMSE lower is better. The LSTM carries internal state across the lookback window, so it captures the nonlinear, state-dependent catchment memory (antecedent soil moisture, snow, routing) that the linear model structurally cannot represent.

## Wall time (this machine)

- BEFORE: 0.13 s   |   AFTER: 20.86 s

Artifacts: `hydrograph.png`, `before_after_bars.png`, `metrics.json`.
