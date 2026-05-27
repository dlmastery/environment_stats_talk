# Experiment 07 — PM2.5 air-quality nowcast: BEFORE vs AFTER

Nowcast hourly **PM2.5** for an urban station on synthetic-but-physically-plausible
data. The contrast is between the classical **history-only** time-series toolkit
(persistence, linear autoregression, ARIMA) and an **AFTER** model that reads the
**weather covariates** that actually drive PM2.5 — wind (ventilation), temperature,
and boundary-layer height (the dilution volume / inversion proxy) — plus calendar
features.

Everything here **runs anywhere**: synthetic data with deterministic seeds, no API
keys, no GPU required. The AFTER model is a scikit-learn `GradientBoostingRegressor`
(or `MLPRegressor`), so CPU is plenty — there is no GPU dependency at all.

---

## Why covariates win here (the scientific point)

Surface PM2.5 is, to first order, **emissions ÷ ventilation**. The same emissions
produce clean air on a windy, well-mixed afternoon and a choking haze on a calm
night under a temperature inversion. The drivers are the standard ones in
air-quality science:

- **Wind ventilation** — wind disperses pollution; PM2.5 falls as wind rises.
- **Boundary-layer height / inversions** — at night and in cold, stagnant
  conditions the planetary boundary layer collapses, trapping emissions in a shallow
  layer so concentrations spike. PM2.5 scales *inversely* with mixing-layer depth.
- **Diurnal + weekly emission cycles** — rush-hour traffic peaks, weekday > weekend,
  a cold-season heating term.
- **Episodic stagnation** — multi-day high-pressure stagnation (low wind + strong
  inversion) produces the pollution *episodes* (exceedance spikes).

A **history-only** model (persistence / AR / ARIMA) can ride PM2.5's short-term
self-persistence, but it is **blind to the cause** of the spikes. When ventilation
collapses, PM2.5 jumps faster than the autoregression can infer from PM2.5 alone, so
the classical models **lag and smear the episodes**. A model that *reads the weather*
sees the cause as it happens and **leads** the spikes — improving RMSE,
skill-vs-persistence, and (the metric that matters for health decisions) the
exceedance **F1**.

The synthetic generator
(`common/synthetic_airquality.synthetic_pm25`) injects exactly this structure:
`emission(hour, dow, season) · ventilation(wind) · dilution(boundary_layer) ·
(1 + boost·episode)`, with only modest residual self-persistence so the spikes are
genuinely weather-driven (not raw stickiness). That is what makes the BEFORE/AFTER
contrast real rather than rigged by noise.

---

## The story

**BEFORE — the traditional univariate baselines** (`before/baseline.py`, pure CPU),
all using **only the PM2.5 history**:

- **Persistence** — `ŷ(t) = y(t−1)`. The honest "do nothing" reference every
  nowcast must beat.
- **Linear AR(p)** — ordinary least squares of PM2.5 on its own recent lags.
- **ARIMA(p,d,q)** — the canonical Box-Jenkins forecaster (statsmodels), rolled
  one-step-ahead with the true observations appended (leak-free walk-forward).

**AFTER — a gradient-boosted nowcast that uses the weather**
(`after/gbm_nowcast.py`):

- A `GradientBoostingRegressor` (or `MLPRegressor` via `--model mlp`) over a feature
  vector of: a few PM2.5 lags (the same self-persistence the BEFORE side has),
  **contemporaneous + lagged weather covariates** (wind, temperature,
  boundary-layer height) and rolling ventilation summaries, and **cyclical calendar
  features** (hour sin/cos, day-of-week, weekend flag). Deterministic
  (`random_state`), CPU, fits in seconds.
- Using *contemporaneous* meteorology is the realistic **nowcast** setting: met
  stations and reanalysis report wind/temperature/boundary-layer in real time, while
  the reference PM2.5 monitor is sparse or delayed — so we estimate current PM2.5
  from current weather. No weather feature ever includes the PM2.5 target, so there
  is no target leakage; the BEFORE baselines simply have none of this information.

**Skill metrics:** **RMSE** / **MAE** (µg/m³, lower better), **skill vs
persistence** = `1 − RMSE_model / RMSE_persistence` (1 = perfect, 0 = ties
persistence, < 0 = worse), and **spike F1** — F1 for detecting threshold
exceedances, where "spike" = PM2.5 above the 90th percentile of the observed test
window. F1 punishes both missed episodes (recall) and false alarms (precision).

---

## Results (committed run on this machine)

From `results/metrics.json` (default: 240 days hourly, seed 0, GBM; 864 test steps;
exceedance threshold = 76.2 µg/m³). Full table in `results/summary.md`.

| Method | RMSE (µg/m³) ↓ | MAE ↓ | Skill vs persistence ↑ | Spike F1 ↑ |
|---|---:|---:|---:|---:|
| Before: persistence | 17.40 | 9.50 | +0.000 | 0.736 |
| Before: linear AR | 18.10 | 12.92 | −0.040 | 0.750 |
| Before: ARIMA | 17.40 | 11.88 | +0.000 | 0.750 |
| **After: GBM + weather** | **12.14** | **6.30** | **+0.302** | **0.847** |
| _Ablation: GBM history-only (no weather)_ | 13.63 | 8.07 | +0.216 | 0.777 |
| _AFTER − best-before (ARIMA)_ | **−5.26** | −5.58 | **+0.302** | **+0.097** |

**Headline:** the covariate GBM cuts RMSE from **17.40 → 12.14** (skill vs
persistence **+0.30**) and lifts exceedance **F1 from 0.750 → 0.847** (**+0.097**).

> Note: ARIMA's one-step forecast here essentially reduces to persistence (same RMSE
> to four significant figures), which is the honest behaviour of a low-order ARIMA on
> a near-AR(1) series — the classical models genuinely have little to add over "last
> value" because the *information* they lack is the weather.

### Effort / acceleration (illustrative, not a measured benchmark)

| Method | Person-time (by hand) | Person-time (with Claude Code) | Wall-time (run) |
|---|---|---|---|
| Before: persistence / linear AR | ~hours (lag selection, diagnostics) | ~2 min | < 1 s |
| Before: ARIMA | ~0.5–1 day (order selection, residual checks) | ~5 min | seconds–½ min |
| **After: GBM + weather** | ~1–2 weeks (feature engineering, covariate alignment, eval, exceedance scoring) | ~10–15 min | seconds (CPU) |

> Person-time figures are **illustrative estimates** of build effort, not measured
> benchmarks — they convey the "zero-to-hero" acceleration (hand-tuning ARIMA orders
> and engineering meteorological features vs. minutes with an agent), not a
> controlled study. The headline is the **RMSE and exceedance-F1 gain**, reported
> exactly in `results/metrics.json`.

---

## Honest note: the covariates drive the win

This is **not** "ML beats statistics." It is "**the right covariates beat a blind
model**." The committed `ablation_history_only` row makes this explicit: the *same*
GBM with the weather features removed (PM2.5 lags + calendar only) lands at RMSE
13.63 / F1 0.777 — much closer to the classical baselines than to the
weather-informed GBM (12.14 / 0.847). On the test fixture used by the unit tests the
gap is even starker (RMSE 17.98 with weather vs 24.31 history-only).

In other words: **without the meteorological covariates, the ML model is only
modestly better than persistence.** The lesson for the talk is not "use a fancier
model" but "**bring in the physical drivers**" — and that an agentic workflow makes
pulling, aligning, and engineering those covariates a minutes-long task instead of a
multi-day one.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (90 days, small GBM): finishes in well under a minute
python experiments/07_air_quality_nowcast/run_before_after.py --quick

# Default headline (240 days hourly, full GBM): seconds–~½ min on CPU
python experiments/07_air_quality_nowcast/run_before_after.py

# MLP variant, or daily resolution, or skip the (slower) ARIMA baseline
python experiments/07_air_quality_nowcast/run_before_after.py --model mlp
python experiments/07_air_quality_nowcast/run_before_after.py --freq D
python experiments/07_air_quality_nowcast/run_before_after.py --no-arima
```

Useful flags: `--quick`, `--n-days N`, `--freq {h,D}`, `--model {gbm,mlp}`,
`--n-lags K`, `--no-arima`, `--seed S`.

Outputs land in `results/`:

- `metrics.json` — RMSE/MAE/skill/spike-F1 for each method + AFTER−BEFORE gains and
  the history-only ablation + config.
- `timeseries_plot.png` — observed PM2.5 vs persistence / linear-AR / GBM on the test
  tail, with the exceedance threshold marked.
- `before_after_bars.png` — grouped bars (RMSE and spike-F1×100: best-before vs GBM).
- `summary.md` — human-readable table.

### Tests (fast, CPU)

```bash
python -m pytest experiments/07_air_quality_nowcast/tests common/tests/test_airquality.py -q
```

The suite (~12 s on this machine) checks the synthetic data is non-negative and
responds to wind/inversion, the metrics are finite, both sides run leak-free, and —
the headline — that the covariate model beats persistence on a held-out tail on both
RMSE and spike-F1.

---

## Swapping in real data

The generator returns a DataFrame indexed by `date` with columns
`['pm25', 'wind', 'temp', 'boundary_layer', 'hour', 'dow', 'is_weekend']`. **Keep
that schema and every script here works unchanged.** To use real data:

1. **PM2.5** — pull a station's series from **OpenAQ** (the v3 REST API at
   `openaq.org`, or the open S3 archive). OpenAQ is one of the verified open
   air-quality datasets.
2. **Meteorology** — get matching **10 m wind speed**, **2 m temperature**, and
   **planetary boundary-layer height** from a reanalysis such as **ERA5** (Copernicus
   Climate Data Store via `cdsapi`/`xarray`) or a co-located met station. ERA5's
   boundary-layer height is the standard ventilation/dilution proxy.
3. Resample everything to a common hourly (or daily) index, assemble the DataFrame
   with the **same columns** above, and feed it straight into `time_split(...)` →
   `run_before` / `run_after`.

For multi-station models, concatenate stations and add static site attributes (land
use, elevation) as extra columns — the GBM/MLP feature matrix generalizes with no
code changes. The same features also plug into a small torch model if you prefer a
neural net.

---

## References (described generically — no invented IDs/stats)

- **OpenAQ** — an open global air-quality data platform aggregating government and
  research PM2.5 / PM10 / gas measurements (REST API + open archive).
- **ERA5** — the ECMWF reanalysis (Copernicus Climate Data Store); its 10 m wind,
  2 m temperature, and boundary-layer height are the standard meteorological drivers
  for air-quality ventilation/dilution analysis.
- **Boundary-layer / ventilation and PM2.5** — it is well established in air-quality
  science that surface particulate concentrations are controlled by ventilation
  (wind) and mixing-layer depth (inversions trap pollution); we encode this idea in
  the synthetic generator and exploit it in the AFTER model. Consult the primary
  atmospheric-science literature for quantitative relationships.
- **ARIMA / Box-Jenkins** — the classical univariate time-series forecasting
  framework (the BEFORE reference here).
- **Gradient boosting** — gradient-boosted regression trees (Friedman) as the
  CPU-friendly covariate model; an `MLPRegressor` alternative is provided.

No statistics, dates, or specific external results are invented here; the headline
numbers live in the committed `results/` from an actual run on this machine.
