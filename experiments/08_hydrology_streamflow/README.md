# Experiment 08 — Rainfall-runoff streamflow forecasting: BEFORE vs AFTER

Predict daily **streamflow** for a catchment from its meteorological forcing
(precipitation, temperature, potential evapotranspiration) on synthetic but
physically-plausible data. This is the **classic case where machine learning
genuinely beats classical hydrology** — so it balances the repo's honest "roughly
tied" experiments with a clean **AFTER-wins** example.

Everything here **runs anywhere**: synthetic data with deterministic seeds, no API
keys, no GPU required. The LSTM auto-uses CUDA (the repo's RTX 4090) when present
and falls back to CPU otherwise.

---

## Why ML wins here (the scientific point)

Real rainfall-runoff has **nonlinear, state-dependent memory**:

- **Antecedent soil moisture** — the same rainfall produces little runoff on dry
  soil (it soaks in) but a lot on already-saturated soil (it runs off). The
  conversion of rain to flow depends *multiplicatively* on a hidden storage state.
- **Snow** — winter precipitation can fall as snow, sit in a snowpack, and release
  days or weeks later when temperatures rise (degree-day melt).
- **Routing** — water takes time to travel to the gauge, so the hydrograph lags and
  recesses smoothly rather than tracking rain instantaneously.

A **linear** regression of streamflow on same-day and lagged precipitation,
`Q = a₀ + Σₖ aₖ·P(t−k) + …`, cannot represent that conditional, multiplicative
storage. It captures the average response but **structurally underfits** the
nonlinear threshold and memory. An **LSTM** reads a lookback window and carries an
internal **cell state** across it — exactly the mechanism needed to track "how full
is the catchment right now" — so it learns the nonlinear memory the linear model
misses. This mirrors the well-established result in operational hydrology that
LSTM rainfall-runoff models outperform calibrated conceptual/regression baselines on
large-sample benchmarks (see *References*).

The synthetic generator (`common/synthetic_hydrology.synthetic_catchment`) is a
minimal **conceptual bucket / soil-moisture model** (HBV / GR4J-family in spirit):
a nonlinear saturation-excess runoff curve `(S/Sₘₐₓ)^β`, a degree-day snow module,
a slow baseflow reservoir, and a triangular routing kernel. It deliberately injects
the nonlinear memory so the BEFORE/AFTER contrast is real, not rigged by noise.

---

## The story

**BEFORE — the traditional linear baseline** (`before/linear_baseline.py`, pure CPU):

- **Linear regression on lagged precipitation** (same-day + lags 0…7), plus an
  **Antecedent Precipitation Index** (exponentially-weighted past rain — the
  classic hydrologist's wetness proxy) and same-day temperature/PET. Fit in closed
  form by ordinary least squares. Transparent and instant — but a fixed linear map.
- **Conceptual single linear-reservoir bucket** (`bucket_forecast`) — the simplest
  classical alternative, with its recession constant calibrated by a small grid
  search on train. Included to show the limitation is the *linear* structure, not
  regression per se.

**AFTER — an LSTM rainfall-runoff model** (`after/lstm_runoff.py`):

- A compact LSTM (1 layer, ~32 hidden) over a `lookback`-day window of
  `[precip, temperature, pet]` + sin/cos day-of-year, standardized on **train only**
  (no leakage), trained with **AdamW + gradient clipping + early stopping** on the
  validation set. Predictions clipped at zero (negative flow is unphysical).
  Deterministic seeds.
- **CPU for the fast tests; auto-CUDA for the headline run** via `get_device()`.

**Skill metrics** (the standard streamflow trio): **NSE** (Nash-Sutcliffe
Efficiency), **KGE** (Kling-Gupta Efficiency) — both `1 = perfect`, higher is
better — and **RMSE** (lower is better). NSE > 0 means "better than predicting the
long-term mean flow"; NSE/KGE in the 0.7–0.9 range is a genuinely skillful model.

---

## Comparison table

Person-time is the realistic hand-build effort for a hydrologist/environmental
statistician vs. the same artifact produced *with Claude Code*. Metric cells are
filled by the committed run in `results/` (see `results/summary.md` for the exact
numbers from the latest run on this machine).

| Method | Person-time (by hand) | Person-time (with Claude Code) | Wall-time (run) | NSE | KGE | Artifacts |
|---|---|---|---|---|---|---|
| Before: linear (lagged precip + API) | ~0.5–1 day (feature/lag selection, diagnostics) | ~2 min | < 0.1 s | see `summary.md` | see `summary.md` | `metrics.json` |
| Before: conceptual bucket | ~days (model choice + manual calibration) | ~5 min | < 1 s | see `summary.md` | see `summary.md` | `metrics.json` |
| **After: LSTM** | ~1–2 weeks (dataset, training loop, tuning, eval) | ~10–15 min | seconds–minutes (CPU) / sub-second/epoch (4090) | **see `summary.md`** | **see `summary.md`** | `metrics.json`, `hydrograph.png`, `before_after_bars.png` |

> Person-time figures are **illustrative estimates** of build effort, not measured
> benchmarks — they convey the "zero-to-hero" acceleration, not a controlled study.
> The headline of this experiment is the **NSE/KGE gain** of the LSTM over the linear
> model, which is reported exactly in `results/metrics.json`.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (tiny series, few epochs): finishes in well under a minute
python experiments/08_hydrology_streamflow/run_before_after.py --quick

# Modest default (25 years, early-stopped LSTM): ~1–3 min; AFTER NSE clearly > BEFORE
python experiments/08_hydrology_streamflow/run_before_after.py

# Headline / GPU pass (longer training; auto-uses the 4090)
python experiments/08_hydrology_streamflow/run_before_after.py --epochs 150
```

Useful flags: `--epochs N`, `--quick`, `--n-years N`, `--n-lags K`, `--seed S`.

Outputs land in `results/`:

- `metrics.json` — NSE/KGE/RMSE for each method + the AFTER−BEFORE gains and config.
- `hydrograph.png` — observed streamflow vs **both** models over the test period.
- `before_after_bars.png` — grouped NSE/KGE bars (linear vs LSTM).
- `summary.md` — human-readable table.

### Tests (fast, CPU)

```bash
python -m pytest experiments/08_hydrology_streamflow/tests common/tests/test_hydrology.py -q
```

The test *logic* is tiny and CPU-only (a few synthetic years, hidden≤24 LSTM,
≤40 capped epochs). The headline test asserts the LSTM beats the linear baseline on
a held-out test period (`NSE_lstm > NSE_linear`). Cold-interpreter wall time is
dominated by torch's one-time import / first-op CPU init, not the assertions.

---

## Swapping in real CAMELS data

The synthetic generator returns a DataFrame indexed by `date` with columns
`['precip', 'temperature', 'pet', 'streamflow']`. **Keep that schema and every
script here works unchanged.** To use real data, swap in **CAMELS** (Catchment
Attributes and Meteorology for Large-sample Studies — the US CAMELS of Newman et
al. / Addor et al., or regional twins CAMELS-GB, -CL, -BR, -AUS), the standard
benchmark behind the published LSTM-rainfall-runoff studies:

1. Download a basin's daily forcings (precipitation, temperature, PET) and observed
   discharge from the CAMELS release.
2. Assemble a `pandas` DataFrame with the **same columns** above, indexed by a
   `date` `DatetimeIndex` (one row per day).
3. Feed it straight into `time_split(...)` → `run_before` / `run_after`.

For multi-basin training (the setting where LSTMs shine most), concatenate basins
and add static catchment attributes as extra input features — the LSTM input tensor
generalizes to `(batch, lookback, n_features)` with no code changes to the model.

---

## Scaling on the RTX 4090

The AFTER model is intentionally small for runs-anywhere reproducibility; the 4090
lets you scale it without changing the code:

- **Auto-device:** `get_device()` returns CUDA automatically — no flag needed.
- **Bigger model / longer memory:** raise `hidden`, `num_layers`, and `lookback`
  (e.g. 128–256 hidden, 2 layers, 180–365-day windows to capture seasonal storage
  and snowpack). Single-basin daily training is tiny for a 4090 (sub-second/epoch).
- **Multi-basin / regional models:** the real LSTM-rainfall-runoff win comes from
  training one model across hundreds of CAMELS basins with static attributes — that
  is a larger but still very tractable workload on a 4090.

---

## References (described generically — no invented IDs/stats)

- **LSTM rainfall-runoff modelling.** A body of operational-hydrology work (Kratzert
  and colleagues, and the broader community) has shown that LSTM models trained on
  large-sample streamflow datasets match or exceed calibrated conceptual/regression
  baselines, especially in a regional multi-basin setting. We describe the *idea*
  here; consult the primary literature for exact datasets, metrics, and numbers.
- **CAMELS** — Catchment Attributes and Meteorology for Large-sample Studies: the
  standard large-sample hydrology benchmark (US CAMELS plus regional counterparts).
- **NSE / KGE** — Nash & Sutcliffe (1970) river-flow forecasting efficiency; Gupta
  et al. (2009) Kling-Gupta Efficiency decomposition (correlation, bias, variability).
- **Conceptual hydrological models** — HBV and GR4J families inspire the synthetic
  bucket/soil-moisture process used to generate the data.

No statistics, dates, or specific results are invented here; the headline numbers
live in the committed `results/` from an actual run on this machine.
