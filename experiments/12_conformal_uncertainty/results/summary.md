# Experiment 12 — Conformal uncertainty: BEFORE vs AFTER

- Point forecaster: ridge on lag features + sin/cos day-of-year (ridge alpha=1.0, lookback=7)
- Data: 20 years synthetic daily temperature, seed 0, heteroscedastic regime {'heat_extreme_rate': 0.06, 'warming_c_per_decade': 0.6}
- Split: chronological 60/20/20 train/calibration/test (no shuffle) | cal targets=1453 | test targets=1453
- Quick mode: False

## Coverage and interval width by nominal level

| Nominal | Method | Empirical coverage | |gap| | Mean width (°C) |
|--------:|--------|-------------------:|-----:|----------------:|
| 80% | before: normal-theory | 0.869 | 0.069 | 6.372 |
| 80% | **after: split conformal** | **0.791** | 0.009 | 5.217 |
| 80% | **after: normalized conformal** | **0.813** | 0.013 | 5.489 |
| 90% | before: normal-theory | 0.929 | 0.029 | 8.178 |
| 90% | **after: split conformal** | **0.904** | 0.004 | 7.269 |
| 90% | **after: normalized conformal** | **0.897** | 0.003 | 7.360 |
| 95% | before: normal-theory | 0.950 | 0.000 | 9.745 |
| 95% | **after: split conformal** | **0.950** | 0.000 | 9.724 |
| 95% | **after: normalized conformal** | **0.943** | 0.007 | 9.256 |

## Mean absolute calibration gap (averaged over levels — lower is better)

| Method | Mean |empirical − nominal| |
|--------|-------------------------:|
| before: normal-theory | 0.0329 |
| after: split conformal | 0.0044 |
| after: normalized conformal | 0.0077 |

Wall time: before 0.197s, after 0.291s.

_Conformal gives finite-sample, distribution-free MARGINAL coverage under exchangeability; it does NOT guarantee conditional (per-season) coverage. The normalized variant only improves conditional coverage._

Artifacts: `metrics.json`, `coverage_plot.png`, `interval_plot.png`.
