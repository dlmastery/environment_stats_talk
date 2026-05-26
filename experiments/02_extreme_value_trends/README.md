# Experiment 02 — Climate extremes & trend detection (BEFORE vs AFTER)

**Task.** Given a daily precipitation record, detect trends and characterise
extremes: annual maxima (Rx1day), the ETCCDI indices Rx5day and R95p, a
Mann-Kendall trend test with Theil-Sen (Sen's) slope, and return levels for the
20-, 50- and 100-year events.

**Data.** Fully synthetic and deterministic — `common.daily_precipitation`
generates an intermittent series (many dry days, seasonally modulated wet-day
probability) with a heavy upper tail whose scale *intensifies over time*. That
injected intensification is the signal both workflows must recover. No API keys,
no GPU, CPU-only.

---

## The before/after story

**BEFORE — the manual EDA day** (`before/manual_eda.py`). What a statistician
codes by hand over the better part of a day: group by year for the annual
maxima, roll-and-group for Rx5day, a fixed 95th-percentile threshold for R95p,
a from-scratch Mann-Kendall test + Sen's slope (no niche package), and a return
level read straight off the **empirical** quantiles. Honest, but limited: point
estimates only, **no distribution fit, no confidence intervals, no validation**.
Crucially, an empirical quantile *cannot extrapolate past the observed record* —
the 100-year level just saturates near the largest annual maximum seen.

**AFTER — the rigorous pipeline in minutes** (`after/agentic_pipeline.py`). The
analysis you actually want to ship, generated and wired together by Claude Code:

- **Block-maxima GEV fit (MLE).** A direct negative-log-likelihood minimisation
  with `scipy.optimize` (vectorised GEV log-density in the genextreme
  parameterisation), with `scipy.stats.genextreme.fit` as a fallback/cross-check.
- **Return levels with bootstrap CIs.** Non-parametric resampling of the block
  maxima, refit each draw, percentile CIs per return period. The GEV
  *extrapolates* to the 100-year level and the CI honestly *widens* with the
  return period.
- **Trend tests with significance**, applied to all three indices (annual
  maxima, Rx5day, R95p), assembled into a results table.
- **A human-in-the-loop `validate()` gate** encoding the checks an
  environmetrician must sign off on (see below).

The shared index/MK definitions live in `before/manual_eda.py` and are
*imported* by the AFTER script, so the only thing that differs is the rigor the
agent adds — not the question being asked.

### The validation gate (`validate()`)

AI runs the checks; the analyst keeps the final call (the project's
statistical-rigor rule). The gate emits advisory warnings + notes and a
non-blocking `passed` flag for:

1. **n-per-block adequacy** — GEV asymptotics are shaky with few blocks; warns
   below ~20 annual maxima.
2. **Stationarity caveat** — a *significant* MK trend in the block maxima
   violates the stationary-GEV assumption, so the return levels are a
   record-average, not a current-climate estimate (points to a non-stationary
   GEV with time-varying location/scale).
3. **Autocorrelation warning** — Mann-Kendall assumes independence; lag-1
   autocorrelation above a threshold inflates the false-positive rate (suggests
   pre-whitening / block bootstrap).
4. **Shape-parameter sanity** — `|xi| > 0.5` on a short record is fragile.
5. **Multiple-testing note** — testing several indices inflates family-wise
   error; reports the Bonferroni-adjusted threshold (alpha / n_tests).

---

## Comparison table

| Dimension | BEFORE (manual EDA) | AFTER (Claude Code + EVT) |
|---|---|---|
| Effort | ~a day of hand-coding | minutes to wire up + run |
| Return level | empirical quantile (cannot extrapolate past record) | GEV MLE, extrapolates to 100-yr+ |
| Uncertainty | none | bootstrap 95% CI on every level |
| Trend | one point Sen's slope | MK z + p for 3 indices, results table |
| Assumption checks | none | stationarity, autocorrelation, n-blocks, shape, multiple-testing |
| Reproducibility | ad hoc | deterministic seed, committed `metrics.json` + plots |
| Rigor preserved | — | the human signs off on the `validate()` warnings before reporting |

**What rigor is preserved.** AFTER is faster, not looser: the index definitions
and Mann-Kendall statistic are *identical* to BEFORE (shared code), the GEV is a
standard MLE cross-checked against scipy, the CIs are an honest bootstrap, and
the validation gate forces the same caveats a careful reviewer would raise
(stationarity, independence, multiple testing). The acceleration is in the
plumbing and breadth of checks, not in cutting statistical corners.

---

## How to run

Run **from the repo root** so `import common` resolves.

```bash
# Fast smoke check (short record, fewer bootstrap reps)
python experiments/02_extreme_value_trends/run_before_after.py --quick

# Full run (writes the committed artifacts)
python experiments/02_extreme_value_trends/run_before_after.py

# Each side standalone
python experiments/02_extreme_value_trends/before/manual_eda.py
python experiments/02_extreme_value_trends/after/agentic_pipeline.py

# Tests (fast, < 30s)
python -m pytest experiments/02_extreme_value_trends/tests -q
```

### Outputs (`results/`)

- `return_levels.png` — GEV return-level curve with a 95% bootstrap CI band and
  observed block maxima on Gringorten plotting positions.
- `trend_plot.png` — annual maxima time series with the Sen's-slope line.
- `metrics.json` — trend slopes & p-values (all indices), GEV parameters, and
  20/50/100-yr return levels with CIs (AFTER GEV + BEFORE empirical).
- `summary.md` — a short before/after readout including the validation gate.

Flags: `--quick`, `--n-years`, `--seed`, `--intensification`.

---

## Swapping in real data (ERA5 / station)

The synthetic generator documents the swap (`common.synthetic_climate.REAL_DATA_NOTE`).
In short:

1. **ERA5 (reanalysis).** `pip install cdsapi xarray`, register at the Copernicus
   Climate Data Store, download `total_precipitation`, and load with xarray.
   Aggregate to a daily total at your grid cell / region.
2. **Station data.** Any daily-precip table (e.g. GHCN-Daily `PRCP`) works.
3. Reduce to the same schema — a `date` index plus a `precip` column — and every
   script here runs unchanged: `annual_maxima`, `rx5day`, `r95p`, `mann_kendall`,
   `fit_gev`, `bootstrap_return_levels`, and `validate` are all data-agnostic.

Real-data caveats the gate already surfaces: check for **non-stationarity** (a
warming-climate trend in the maxima often warrants a non-stationary GEV), for
**serial dependence** in the blocks, and for **record length** (return levels far
beyond the record length are extrapolations — report the CI, not just the point).

---

## Methods & references (generic EVT literature)

Standard, verifiable methods only — no invented papers or statistics. For exact
citations, **(verify)** against the primary sources before putting them on a slide:

- **Extreme Value Theory / GEV & block maxima.** Coles, *An Introduction to
  Statistical Modeling of Extreme Values* (Springer) is the standard textbook
  reference for the GEV, block-maxima MLE, return levels, and profile/bootstrap
  uncertainty. **(verify edition/year)**
- **Mann-Kendall trend test.** The non-parametric rank-based trend test
  (Mann; Kendall) as described in standard environmetrics references. **(verify)**
- **Theil-Sen / Sen's slope.** The median-of-pairwise-slopes estimator
  (Theil; Sen). **(verify)**
- **ETCCDI extreme indices** (Rx1day, Rx5day, R95p) — the WMO/ETCCDI climate
  extremes index definitions used here. **(verify exact definitions)**
- **Gringorten plotting position** for empirical return periods. **(verify)**

These are well-established, textbook methods; the implementations here are
self-contained (scipy/statsmodels only) and cross-checked, so the results do not
depend on any single niche package.
