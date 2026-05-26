# Experiment 01 — Climate time-series forecasting: BEFORE vs AFTER

Forecast daily **2 m air temperature** `h` steps ahead on synthetic, ERA5-like
data. This is the "hello world" of environmental forecasting and the cleanest
place to show the **before/after** thesis of the talk: the *traditional*
statistician's toolkit versus a *small deep model stood up in minutes with Claude
Code* — plus a pointer to the zero-shot **foundation-model** frontier.

Everything here **runs anywhere**: synthetic data (deterministic seeds), no API
keys, no GPU required. The model auto-uses CUDA (the repo's RTX 4090) when present.

---

## The story

**BEFORE — the honest classical baselines** (`before/baseline.py`, pure CPU):

- **Persistence / random walk** — "tomorrow ≈ today". The canonical reference for
  forecast *skill*; at `h = 1` day it is genuinely hard to beat because daily
  temperature is highly autocorrelated.
- **Seasonal-naive** — "this day next-period ≈ this day last year (365 d ago)".
  Captures the annual cycle for free; competitive at long horizons, weak at short.
- **(S)ARIMA / AutoReg** — a classical autoregressive fit via `statsmodels`, rolled
  forward out-of-sample. Interpretable, but slow to tune and brittle to set up.
  (The script prefers `SARIMAX`; if the environment's statespace extensions are
  unavailable it transparently falls back to `AutoReg`, an AR(p) on the de-trended
  signal — same classical idea, always runnable.)

**AFTER — a small LSTM, built in minutes** (`after/dl_forecaster.py`):

- A compact LSTM (1 layer, ~48 hidden) over a windowed dataset (past `lookback`
  days + sin/cos day-of-year), standardized on **train only**, trained with
  **AdamW + early stopping** on validation RMSE. Deterministic seeds.
- **CPU for tests; auto-CUDA for the headline run** via `get_device()`.
- **OPTIONAL** `zero_shot_foundation_baseline()` documents how to drop in a real
  time-series foundation model — Amazon **Chronos** (`chronos-forecasting`) or
  Google **TimesFM** (`timesfm`) — for a *zero-shot* forecast with no task-specific
  training. It is skipped cleanly if the package is absent, so nothing here ever
  requires a download or key.

**What the comparison teaches (for a skeptical audience):** at `h = 1` a deep model
should be *roughly tied with* persistence — and that is the point. The win shows up
at longer horizons and on richer, multivariate, multi-station data where the
classical baselines plateau; the foundation-model row shows the new "no-training"
option. The skill score (`1 − RMSE_after / RMSE_persistence`) keeps everyone honest.

---

## Comparison table

Person-time is the realistic hand-build effort for an environmental statistician
vs. the same artifact produced *with Claude Code*. Wall-time is compute on this
machine. Metric cells are **placeholders** until the full GPU pass is run (the
committed `results/` currently reflect a fast `--quick` smoke run).

| Method | Person-time (by hand) | Person-time (with Claude Code) | Wall-time (run) | Test RMSE (°C) | Artifacts |
|---|---|---|---|---|---|
| Persistence | ~0.5 h | ~1 min | < 0.1 s | _placeholder_ | `metrics.json` |
| Seasonal-naive | ~1 h | ~1 min | < 0.1 s | _placeholder_ | `metrics.json` |
| (S)ARIMA / AutoReg | ~0.5–1 day (order selection, diagnostics) | ~2 min | seconds (capped) | _placeholder_ | `metrics.json` |
| LSTM (AFTER) | ~1–2 days (dataset, training loop, early stop, eval) | ~5–10 min | seconds (CPU) / sub-second/epoch (4090) | _placeholder_ | `metrics.json`, `forecast_plot.png`, `before_after_bars.png` |
| Foundation zero-shot (Chronos/TimesFM) | ~days (find, install, wire up) | ~5 min (optional, skipped if absent) | seconds (zero-shot) | _optional_ | `metrics.json` (row when enabled) |

> Person-time figures are **illustrative estimates** of build effort, not measured
> benchmarks — they convey the "zero-to-hero" acceleration, not a controlled study.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (tiny series, few epochs, SARIMA off): finishes in well under a minute
python experiments/01_climate_timeseries_forecast/run_before_after.py --quick

# Modest default (20 years, h = 1 and 7, early-stopped LSTM): < ~2 min on CPU
python experiments/01_climate_timeseries_forecast/run_before_after.py

# Headline / GPU pass (longer training; auto-uses the 4090)
python experiments/01_climate_timeseries_forecast/run_before_after.py --epochs 80
```

Useful flags: `--epochs N`, `--quick`, `--n-years N`, `--horizons 1 7 14`,
`--seed S`, `--no-foundation`.

Outputs land in `results/`:

- `metrics.json` — RMSE/MAE/ACC per method & horizon + skill of AFTER vs persistence.
- `forecast_plot.png` — observed vs persistence vs LSTM on the test tail.
- `before_after_bars.png` — grouped RMSE bars (persistence vs LSTM).
- `summary.md` — human-readable table.

### Tests (fast, CPU)

```bash
python -m pytest experiments/01_climate_timeseries_forecast/tests -q
```

The test *logic* is tiny and CPU-only (2-year series, 1 epoch, 10-point AR fit).
Cold-interpreter wall time is dominated by one-time library start-up (importing
`torch` and `statsmodels`, plus torch's first-op CPU init) rather than the
assertions themselves.

---

## Swapping in real ERA5 data

The synthetic generator (`common/synthetic_climate.daily_temperature`) returns a
DataFrame indexed by `date` with a single `t2m` column. **Keep that schema and
every script here works unchanged.** To use real ERA5 2 m temperature:

```bash
pip install cdsapi xarray netCDF4
```

1. Register at the Copernicus **Climate Data Store** and place your API key in
   `~/.cdsapirc`.
2. Download `2m_temperature` (hourly or daily) for a station/grid cell and period.
3. Load and reduce to a daily series with `xarray`, then feed the resulting
   `t2m` DataFrame straight into `time_split(...)` → `run_before` / `run_after`:

```python
import xarray as xr
ds = xr.open_dataset("era5_t2m.nc")
t2m = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")
daily = (t2m.resample(time="1D").mean() - 273.15)  # K -> °C
df = daily.to_dataframe(name="t2m")[["t2m"]].rename_axis("date")
```

For *gridded* forecasting use latitude-weighted RMSE/ACC
(`common.metrics.latitude_weighted_rmse`) — the convention used by ERA5-based
weather-model benchmarks.

---

## Scaling on the RTX 4090

The AFTER model is intentionally small for runs-anywhere reproducibility; the
4090 lets you scale it without changing the code:

- **Auto-device:** `get_device()` returns CUDA automatically — no flag needed.
- **Bigger model / longer context:** raise `hidden`, `num_layers`, and `lookback`
  (e.g. 256 hidden / 2–3 layers / 90–180 day windows); increase `batch_size` to
  saturate the GPU. Daily-resolution single-station training is tiny for a 4090 —
  sub-second per epoch — so spend the budget on more years, more stations
  (multivariate), and more horizons.
- **Foundation models:** Chronos/TimesFM checkpoints fit comfortably in 16–17 GB
  and the 4090 makes zero-shot forecasting interactive; enable
  `zero_shot_foundation_baseline()` after `pip install chronos-forecasting`.
- **Mixed precision / batching:** for large grids, wrap training in
  `torch.autocast("cuda")` and batch across grid cells.

---

## Verifiable anchors

- **ClimateLLM** — arXiv:2502.11059 (LLMs for climate tasks; cited as a real,
  verifiable reference per the project's zero-hallucination rule).
- **Time-series foundation models** used in `zero_shot_foundation_baseline()`:
  Amazon **Chronos** and Google **TimesFM** — both real, publicly released models.

No statistics, dates, or results are invented here; the metric cells above are
explicit placeholders until the committed full run is produced on the 4090.
