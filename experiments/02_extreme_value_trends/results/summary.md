# Experiment 02 — Extreme-value & trend detection: BEFORE vs AFTER

Synthetic daily precipitation, 40 years, seed 0, intensification 0.08/decade.

## Trend (annual maxima, Rx1day)

- Mann-Kendall: z = +3.27, p = 0.0011 over 40 annual blocks.
- Sen's slope: +0.3522 mm/yr.

## Return levels (mm)

| T (yr) | BEFORE (empirical) | AFTER (GEV) | AFTER 95% CI |
|---:|---:|---:|---:|
| 20 | 45.19 | 46.73 | [42.34, 52.08] |
| 50 | 47.33 | 51.94 | [45.19, 65.53] |
| 100 | 48.63 | 55.75 | [46.68, 80.22] |

## GEV fit

- Method: nll-min; mu = 29.18, sigma = 6.16, xi = -0.028 (shape c = 0.028), n = 40 blocks.

## Trend table (all indices)

| Index | Sen slope (mm/yr) | MK z | MK p |
|---|---:|---:|---:|
| AnnualMaxima(Rx1day) | +0.3522 | +3.27 | 0.0011 |
| Rx5day | +0.2845 | +1.97 | 0.0490 |
| R95p | +2.7183 | +3.39 | 0.0007 |

## Human-in-the-loop validation

- Passed (no blocking warnings): **False**
- [WARN] Annual maxima show a statistically significant trend (MK p=0.0011): the stationary-GEV assumption is violated. The reported return levels are a record-average; for a current-climate estimate fit a non-stationary GEV (time-varying location/scale).
- [note] Block count adequate: 40 annual maxima (>= 20).
- [note] Block-maxima lag-1 autocorrelation 0.08: independence OK.
- [note] GEV shape xi=-0.03 in a plausible range (|xi|<=0.5).
- [note] Multiple-testing: 3 trend tests reported; use a Bonferroni threshold of alpha/3 = 0.0167 for family-wise control.

## Artifacts

- `return_levels.png`: `C:/Users/evija/climate/environment_stats_talk/experiments/02_extreme_value_trends/results/return_levels.png`
- `trend_plot.png`: `C:/Users/evija/climate/environment_stats_talk/experiments/02_extreme_value_trends/results/trend_plot.png`
- `metrics.json`: `C:/Users/evija/climate/environment_stats_talk/experiments/02_extreme_value_trends/results/metrics.json`

## What AFTER adds over BEFORE

- A fitted GEV (MLE) instead of an empirical quantile read-off.
- Bootstrap confidence intervals on every return level.
- Significance and a multi-index trend table, not just a point slope.
- A validation gate (stationarity, autocorrelation, block-count, shape sanity, multiple-testing) the analyst must sign off on.