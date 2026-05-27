# Tutorial — Run an Experiment End-to-End (Experiment 01: Climate Forecasting)

This is a deep, narrated walkthrough of a single experiment, from the synthetic
data all the way to the honest finding and the path to real ERA5. We use
**Experiment 01 — daily temperature forecasting** because it is the cleanest place
to see the talk's whole thesis: the *traditional* environmental-statistician's
toolkit (persistence, seasonal-naive, SARIMA) versus a *small deep model stood up
in minutes with Claude Code* (an LSTM), all built, run, and skill-scored in one
command — and an honest result where **classical statistics wins at the longer
lead times.**

Everything here runs anywhere: synthetic data, deterministic seeds, no API keys,
CPU fine (the LSTM auto-uses CUDA if a GPU is present). If you have not set up the
repo yet, do `docs/GETTING_STARTED.md` first.

> **Run from the repo root.** Every command below assumes your working directory is
> the `environment_stats_talk` folder so that `import common` resolves.

---

## 1. The question

> *Forecast daily 2 m air temperature `h` days ahead, and find out — fairly and
> fast — whether a small neural net actually beats the classical baselines, and at
> which lead times.*

The forecast **lead time** (or horizon) `h` is how many days ahead we predict. We
evaluate at `h = 1, 7, 14` days. The reason to sweep `h` is that the answer
*changes with horizon*, and that change is the most important lesson of the
experiment.

---

## 2. The synthetic data (what we are forecasting, and why it is honest)

The data comes from `common.synthetic_climate.daily_temperature` — a deterministic
generator that produces an **ERA5-like** daily 2 m temperature series: a DataFrame
indexed by `date` with a single `t2m` column. The signal is built from the physical
ingredients you would expect in a real station record:

- a strong **annual cycle** (the seasonal sinusoid — warm summers, cold winters);
- **day-to-day autocorrelation** (today's temperature is highly predictive of
  tomorrow's — this is what makes persistence such a strong short-horizon baseline);
- a slow **warming trend**;
- **noise** and a heavy-tailed **heat-extreme** component.

Two properties make the resulting before/after comparison *honest rather than
rigged*:

1. **Determinism.** With a fixed seed the series is byte-for-byte reproducible, so
   the committed numbers are exactly reproducible with one command.
2. **High autocorrelation at short lags.** Because daily temperature is nearly an
   AR(1) process, "tomorrow ≈ today" (persistence) is *genuinely hard to beat at
   `h = 1`*. The generator does not hand the deep model an easy win — any
   improvement has to be earned, and it shows up mostly at longer horizons.

The default full run uses **20 years** of daily data, split chronologically (never
shuffled — that would leak the future of a warming series) into roughly
**5110 train / 1095 validation / 1095 test** days.

---

## 3. BEFORE — the honest classical baselines

These live in `before/baseline.py`, are pure CPU, and represent what a careful
environmental statistician reaches for first.

### 3.1 Persistence (random walk)

`ŷ(t + h) = y(t)` — "the forecast for `h` days from now is today's value." This is
the canonical reference for forecast **skill**: any model that cannot beat
persistence has added nothing. At `h = 1` it is a very strong baseline precisely
because daily temperature is so autocorrelated.

### 3.2 Seasonal-naive

`ŷ(t + h) = y(t + h − 365)` — "this calendar day next period looks like this
calendar day last year." It captures the annual cycle for free, so it is
competitive at *long* horizons (where persistence has decayed) but weak at short
ones. Its RMSE is flat across horizons (3.71 at every lead in the committed run)
because it always reaches back a full year regardless of `h`.

### 3.3 (S)ARIMA

A classical autoregressive fit via `statsmodels`, rolled forward out-of-sample. The
script prefers `SARIMAX` (the seasonal state-space model); if the environment's
state-space extensions are unavailable it transparently falls back to `AutoReg`
(an AR(p) on the de-trended signal) so it always runs. SARIMA is interpretable and,
as we will see, **very strong here** — but in real practice it is slow to tune
(order selection, residual diagnostics) and brittle to set up. That hand-tuning
cost is exactly what the agentic workflow compresses.

---

## 4. AFTER — a small LSTM, built in minutes

This lives in `after/dl_forecaster.py`. The "AFTER" claim is not "a neural net is
magic" — it is *"Claude Code wrote, trained, and benchmarked this in minutes, and
we can compare it fairly to the classical baselines in the same run."*

The model is intentionally **compact**: a 1-layer LSTM with ~48 hidden units over a
windowed dataset (the past `lookback = 30` days plus `sin/cos` day-of-year as
seasonal features), standardized on **train only** (no leakage), trained with
**AdamW + early stopping** on validation RMSE, with deterministic seeds. It runs on
CPU for tests and **auto-uses CUDA** for the headline run via `get_device()`.

There is also an **optional** `zero_shot_foundation_baseline()` that documents how
to drop in a real time-series foundation model — Amazon **Chronos** or Google
**TimesFM** — for a zero-shot forecast with *no task-specific training*. It is
skipped cleanly if the package is absent (the committed `metrics.json` records a
`"status": "skipped"` for this row), so nothing here ever requires a download or
key.

---

## 5. The skill score (the number that keeps everyone honest)

For the AFTER model we report:

```
skill_score = 1 − RMSE_after / RMSE_persistence
```

Interpretation:

- **`skill = 0`** — the model ties persistence (added nothing).
- **`skill > 0`** — the model beats persistence; `0.30` means a 30% RMSE reduction
  relative to the naive floor.
- **`skill < 0`** — the model is *worse* than doing nothing.

We score skill against **persistence** specifically because persistence is the
trivial reference every forecaster must beat to justify its existence. Note that
skill-vs-persistence does *not* by itself tell you whether the LSTM beats SARIMA —
for that you compare their RMSEs directly, which is why both appear in the table.

---

## 6. Run it

```bash
# fast smoke run (tiny series, few epochs, SARIMA off) — finishes in well under a minute
python experiments/01_climate_timeseries_forecast/run_before_after.py --quick

# the modest default (20 years, early-stopped LSTM) — ≈ 2 min on CPU; this produced the committed numbers
python experiments/01_climate_timeseries_forecast/run_before_after.py

# headline / GPU pass (longer training; auto-uses the 4090 if present)
python experiments/01_climate_timeseries_forecast/run_before_after.py --epochs 80
```

Useful flags: `--epochs N`, `--quick`, `--n-years N`, `--horizons 1 7 14`,
`--seed S`, `--no-foundation`.

---

## 7. Reading the results

### 7.1 `results/metrics.json` — the machine-readable scoreboard

This is the source of truth. Its structure (committed full run: `n_years = 20`,
`seed = 0`, `device = cuda`):

- `config` — the run settings: years, horizons, seed, the LSTM hyper-parameters
  (`lookback = 30`, `hidden = 48`, `num_layers = 1`, `lr = 0.003`,
  `batch_size = 64`, `epochs = 80`, `patience = 6`), the device used, and the
  train/val/test split sizes (`5110 / 1095 / 1095`).
- `horizons` — one block per lead time, each with a `before` sub-block
  (`persistence`, `seasonal_naive`, `sarima`, each with `rmse` / `mae` / `acc`), an
  `after` sub-block (the LSTM's `rmse` / `mae` / `acc`, plus `epochs_run`, `device`,
  and `best_val_rmse_z`), the `skill_score_after_vs_persistence`, the number of
  evaluation targets (`1095`), and the `foundation_zero_shot` status.
- `wall_time_sec` — the before/after wall-clock per horizon.

The committed numbers:

| Lead `h` | persistence RMSE | seasonal-naive RMSE | **SARIMA RMSE** | **LSTM (AFTER) RMSE** | LSTM skill vs persistence |
|---------:|-----------------:|--------------------:|----------------:|----------------------:|--------------------------:|
| 1 day  | 1.876 | 3.713 | 1.826 | **1.767** | **+0.058** |
| 7 days | 3.580 | 3.713 | **2.054** | 2.811 | +0.215 |
| 14 days | 4.308 | 3.713 | **2.095** | 2.937 | +0.318 |

(All RMSE in °C. The bold cell in each row is the *best* model at that horizon. The
LSTM also stopped early at every horizon — `epochs_run` 31, 27, 7 respectively —
which is the early-stopping doing its job.)

### 7.2 The plots

- **`forecast_plot.png`** — observed temperature vs persistence vs the LSTM on the
  test tail. The seasonal swing dominates the picture; look at how each forecast
  tracks (or lags) the daily wiggles.
- **`before_after_bars.png`** — grouped RMSE bars (persistence vs LSTM), the
  one-glance version of the skill story.

### 7.3 `results/summary.md`

The same scoreboard as a human-readable Markdown table — the artifact you would
drop onto a slide.

---

## 8. The honest finding (this is the whole point for a skeptical audience)

Read the table in §7.1 again, horizon by horizon:

- **At `h = 1` day:** persistence is 1.876, SARIMA 1.826, the LSTM 1.767. The LSTM
  edges ahead (skill +0.058) but **all three are within a hair of each other.** This
  is *expected and correct* — one-step daily temperature is a near-AR(1) problem, so
  "tomorrow ≈ today" is already excellent and there is very little headroom. A deep
  model that merely *ties* persistence at `h = 1` is behaving exactly as theory
  predicts.

- **At `h = 7` and `h = 14`:** the LSTM beats persistence handily (skill +0.215 and
  +0.318 — the headline numbers in `RESULTS.md`: persistence 4.31 → LSTM 2.94 at the
  14-day lead). **But classical SARIMA wins outright**: 2.054 / 2.095 °C beats the
  LSTM's 2.811 / 2.937. The seasonal autoregressive structure of the series is
  precisely what SARIMA is built to exploit, and on a single clean station it does
  so better than a small LSTM.

**So the verdict is: SARIMA wins at ≥ 7-day lead — classical statistics beats the
net here.** That is not a failure of the experiment; it *is* the experiment. The
value the agentic workflow delivers is not "the neural net won" — it is that
**persistence + seasonal-naive + SARIMA + an LSTM were all built and skill-scored in
a single command**, fairly and reproducibly, so you *find out fast* which approach
actually wins instead of assuming. The talk's "before/after" is the collapse in
**human effort** (days of bespoke coding for four methods + plots + skill scores →
one `run_before_after.py` in minutes), not a claim that deep learning dethrones
classical time-series analysis on every problem.

Where would the LSTM be expected to pull ahead? On **richer, multivariate,
multi-station** data where the classical baselines plateau — more covariates, more
spatial structure, more nonlinearity than a single AR-like series can express. The
foundation-model row (Chronos/TimesFM) is the *next* frontier: a competitive
forecast with *no task-specific training at all*. Experiment 01 is the controlled,
honest baseline; the bigger wins live in the harder experiments
(see `docs/EXPERIMENTS_INDEX.md`).

---

## 9. The unit tests

```bash
python -m pytest experiments/01_climate_timeseries_forecast/tests -q
```

The test logic is tiny and CPU-only (a 2-year series, 1 epoch, a 10-point AR fit) —
it checks the pipeline runs end-to-end, the metrics are finite, and the splits are
leak-free, not that the model is accurate. The cold-start wall time you observe is
dominated by one-time library import (`torch`, `statsmodels`), not the assertions.

---

## 10. Scaling up on the RTX 4090

The AFTER model is deliberately small so it runs anywhere; the 4090 lets you scale
**without changing the code**:

- **Auto-device.** `get_device()` returns CUDA automatically — no flag needed; the
  device is recorded in `metrics.json` (the committed run shows `"device": "cuda"`).
- **Bigger model / longer context.** Raise `hidden`, `num_layers`, and `lookback`
  (e.g. 256 hidden / 2–3 layers / 90–180-day windows) and increase `batch_size` to
  saturate the GPU. Daily single-station training is *tiny* for a 4090 — sub-second
  per epoch — so spend the compute budget on **more years, more stations
  (multivariate), and more horizons**, which is exactly the regime where the LSTM is
  expected to overtake the classical baselines.
- **Foundation models.** Chronos / TimesFM checkpoints fit comfortably in the
  4090's ~16–17 GB, making zero-shot forecasting interactive. Enable
  `zero_shot_foundation_baseline()` after `pip install chronos-forecasting`.
- **Mixed precision / batching.** For large grids, wrap training in
  `torch.autocast("cuda")` and batch across grid cells.

---

## 11. Swapping in real ERA5 data

The synthetic generator returns a DataFrame indexed by `date` with a single `t2m`
column. **Keep that schema and every script in this experiment works unchanged.**
To use real ERA5 2 m temperature:

```bash
pip install cdsapi xarray netCDF4
```

1. Register at the Copernicus **Climate Data Store** and put your API key in
   `~/.cdsapirc`.
2. Download `2m_temperature` (hourly or daily) for a station / grid cell and period.
3. Reduce to a daily `t2m` series and feed it straight into
   `time_split(...)` → `run_before` / `run_after`:

```python
import xarray as xr
ds = xr.open_dataset("era5_t2m.nc")
t2m = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")
daily = (t2m.resample(time="1D").mean() - 273.15)  # K -> °C
df = daily.to_dataframe(name="t2m")[["t2m"]].rename_axis("date")
```

For **gridded** forecasting, switch to latitude-weighted RMSE/ACC
(`common.metrics.latitude_weighted_rmse`) — the convention used by ERA5-based
weather-model benchmarks.

---

## 12. What this experiment teaches in one sentence

> The agent compresses *days* of building four forecasters, their plots, and their
> skill scores into *one command in minutes* — and the honest, reproducible result
> it hands you is that **classical SARIMA still wins this problem at lead times of a
> week or more**, which is exactly the kind of fast, fair, unflattering answer a
> rigorous environmental statistician should want.

### Verifiable anchors

- **ClimateLLM** — arXiv:2502.11059 (LLMs for climate tasks).
- Time-series foundation models referenced in
  `zero_shot_foundation_baseline()`: Amazon **Chronos** and Google **TimesFM** —
  both real, publicly released models.

No statistics or dates are invented in this tutorial; every number comes from
`experiments/01_climate_timeseries_forecast/results/metrics.json` and `RESULTS.md`.
