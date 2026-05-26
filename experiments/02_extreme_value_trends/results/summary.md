# Experiment 02 — Extreme-value & trend detection: BEFORE vs AFTER

Synthetic daily precipitation, 25 years, seed 1, intensification 0.12/decade (QUICK mode).

## Trend (annual maxima, Rx1day)

- Mann-Kendall: z = +2.08, p = 0.0377 over 25 annual blocks.
- Sen's slope: +0.5457 mm/yr.

## Return levels (mm)

| T (yr) | BEFORE (empirical) | AFTER (GEV) | AFTER 95% CI |
|---:|---:|---:|---:|
| 20 | 51.44 | 51.34 | [42.26, 68.09] |
| 50 | 53.29 | 59.86 | [44.54, 96.88] |
| 100 | 53.43 | 66.80 | [46.44, 128.43] |

## GEV fit

- Method: nll-min; mu = 28.95, sigma = 6.46, xi = 0.101 (shape c = -0.101), n = 25 blocks.

## Trend table (all indices)

| Index | Sen slope (mm/yr) | MK z | MK p |
|---|---:|---:|---:|
| AnnualMaxima(Rx1day) | +0.5457 | +2.08 | 0.0377 |
| Rx5day | +0.4018 | +1.38 | 0.1682 |
| R95p | +4.3511 | +1.99 | 0.0471 |

## Human-in-the-loop validation

- Passed (no blocking warnings): **False**
- [WARN] Annual maxima show a statistically significant trend (MK p=0.0377): the stationary-GEV assumption is violated. The reported return levels are a record-average; for a current-climate estimate fit a non-stationary GEV (time-varying location/scale).
- [note] Block count adequate: 25 annual maxima (>= 20).
- [note] Block-maxima lag-1 autocorrelation -0.17: independence OK.
- [note] GEV shape xi=0.10 in a plausible range (|xi|<=0.5).
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