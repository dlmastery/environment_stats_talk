# Experiment 06 — Spatial interpolation: BEFORE vs AFTER

Map an environmental variable (think **temperature**, or a **pollutant**) over a 2-D
domain from a sparse network of **scattered station observations**, then score the
map against a dense held-out grid of the true field. This is the classic
**geostatistics** task — and a clean before/after for the talk:

- **BEFORE** — the traditional **coordinate-only** interpolators a geostatistician
  reaches for: **Inverse-Distance Weighting** and **ordinary kriging** (here as a
  Gaussian process with an RBF covariance + nugget). They model spatial
  autocorrelation from `(x, y)` alone.
- **AFTER** — a **covariate-aware** machine-learning interpolator (RandomForest /
  GradientBoosting) that is also given an auxiliary **covariate** (e.g. elevation)
  at every prediction location, plus a few engineered features.

Everything here **runs anywhere**: synthetic data with deterministic seeds, no API
keys, **CPU-only**, finishes in seconds.

---

## Why covariate-aware ML wins here (the scientific point)

The synthetic field (`common/synthetic_spatial.synthetic_spatial_field`) is built as

```
z(x, y) = trend(x, y)  +  lapse · elevation(x, y)  +  GRF(x, y)
```

with two distinct kinds of structure:

1. A **smooth, spatially-autocorrelated** part — `trend` (a gentle large-scale
   gradient) plus `GRF`, a Gaussian random field with a squared-exponential
   covariance (correlation length ≈ 18 units). Nearby places are similar. **This is
   exactly what kriging was built to model, and it captures it well.**
2. A **covariate-driven** part — `lapse · elevation`, where `elevation` is a
   multi-scale terrain surface (the covariate) that changes on **short** spatial
   scales (a ridge next to a valley). Two stations a few units apart can have very
   different values even though they are "close" in `(x, y)`. A coordinate-only
   interpolator must **smooth across** that, leaving real, *explainable* error on
   the table. A model that is **given the elevation** at the prediction cell
   recovers it directly.

So coordinate-only kriging is a strong, honest baseline (it clearly beats naive IDW),
but it structurally cannot reproduce the short-scale covariate response — which is
the gap the AFTER model closes. The covariate is genuinely informative: on the grid
it explains a large fraction of the field's variance (univariate R² ≈ 0.88 in the
committed run), and the AFTER model's feature importances put most of the weight on
the covariate and its interactions — confirming it is using the covariate, not
re-learning coordinates.

---

## The story

**BEFORE — coordinate-only interpolation** (`before/kriging.py`, pure CPU):

- **IDW** — `ẑ(s₀) = Σ wᵢ zᵢ / Σ wᵢ`, `wᵢ = 1/dᵢᵖ`. Transparent, no fitting, one knob.
- **Ordinary kriging via a GP** — scikit-learn `GaussianProcessRegressor` with a
  `ConstantKernel · RBF + WhiteKernel` (the RBF *is* the Gaussian covariance model
  of geostatistics; its length scale and the nugget are fitted by maximising the
  marginal likelihood — the modern, automatic stand-in for fitting a variogram by
  eye). It also returns a **predictive-variance surface** — kriging's signature.
- **Empirical variogram** (`empirical_variogram`) — the classic Matheron
  semivariogram diagnostic (`variogram.png`): γ(h) rising toward a sill is the
  "yes, there is exploitable spatial autocorrelation" check.

**AFTER — covariate-aware ML** (`after/ml_interp.py`):

- A scikit-learn tree ensemble (**RandomForest** default; **GradientBoosting**
  selectable, plus a small **MLP** alternative) on features
  `[x, y, covariate, x·y, covariate², x·covariate, y·covariate]`. The raw
  coordinates still let it learn the large-scale trend; the covariate and its
  interactions let it learn the lapse response. Deterministic (`random_state=0`),
  fast on CPU, no feature scaling needed for trees.

**Metrics:** RMSE and MAE on the dense held-out grid's **noise-free** truth (so error
is measured against the real field, not noisy re-observations).

---

## Headline result (committed run on this machine)

Default config: 180 stations, a 45×45 = 2025-cell held-out grid, RF (400 trees),
seed 0. Numbers are from `results/metrics.json` / `results/summary.md`.

| Method | Uses covariate? | RMSE ↓ | MAE ↓ |
|---|:---:|---:|---:|
| Before: IDW (coords only) | no | 4.94 | 3.64 |
| Before: ordinary kriging / GP (coords only) | no | 2.91 | 2.24 |
| **After: RF + covariate** | **yes** | **2.20** | **1.60** |
| _AFTER − kriging_ | | **−0.70** | **−0.64** |

**Covariate-aware ML cuts held-out RMSE by ≈ 24% versus coordinate-only kriging**
(skill score `1 − RMSE_ml/RMSE_krig ≈ 0.24`), and kriging in turn clearly beats naive
IDW — so the win is over a *real* geostatistical baseline, not a strawman. (With
GradientBoosting the margin is larger still; see `--model gbm`.)

### Effort / time comparison

Person-time is the realistic hand-build effort for a geostatistician vs. the same
artifact produced *with Claude Code*.

| Method | Person-time (by hand) | Person-time (with Claude Code) | Wall-time (run) | Deliverable |
|---|---|---|---|---|
| Before: IDW | ~1 hr | ~1 min | < 0.1 s | point map |
| Before: ordinary kriging | **~a day** (variogram fitting/diagnostics, model & anisotropy choice, cross-validation) | ~2–3 min | seconds | point map **+ calibrated variance surface** |
| **After: RF/GBM + covariate** | ~1–2 days (feature engineering, covariate alignment, training, eval) | ~5–10 min | seconds | lower-RMSE point map (no variance surface) |

> Person-time figures are **illustrative estimates** of build effort, not measured
> benchmarks — they convey the "zero-to-hero" acceleration. The headline is the
> **RMSE gain from the covariate**, reported exactly in `results/metrics.json`.

---

## Honest note — the uncertainty-surface tradeoff

**This is not a free lunch, and the talk should say so.** Ordinary kriging is not
merely "the old way": it yields a principled, spatially-varying **predictive-variance
map** (largest far from stations) that is exactly what many environmental
deliverables *require* — risk maps, monitoring-network design, data-assimilation
weights, confidence bounds on an exposure estimate. The covariate-aware tree model
here wins on point accuracy but provides **no calibrated uncertainty surface** for
free. Lower RMSE is genuinely not the whole story.

The truly "hero" workflow **combines** both: feed the covariate into a geostatistical
model — **regression-kriging** (kriging the residuals of a covariate regression) or a
**GP whose mean function / inputs include the covariate** — to get *both* the lower
error *and* a calibrated uncertainty surface. The before/after here isolates the
single lever (the covariate) on purpose; the combined method is the natural next step
and is noted in the summary.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (fewer stations, coarse grid, fewer trees): a few seconds
python experiments/06_spatial_interpolation/run_before_after.py --quick

# Default (180 stations, 45×45 grid, RF-400): well under a minute on CPU
python experiments/06_spatial_interpolation/run_before_after.py

# Try GradientBoosting (larger margin) or the MLP backbone
python experiments/06_spatial_interpolation/run_before_after.py --model gbm
python experiments/06_spatial_interpolation/run_before_after.py --model mlp
```

Useful flags: `--quick`, `--model {rf,gbm,mlp}`, `--n-points N`, `--grid-res N`,
`--n-estimators N`, `--seed S`.

Outputs land in `results/`:

- `metrics.json` — RMSE/MAE for IDW, kriging and ML + AFTER-vs-kriging gains, the
  covariate-informativeness numbers, the uncertainty note, and config.
- `predicted_surface.png` — four maps: covariate field, **true** field (with station
  locations), **kriging** map, **ML** map (shared colour scale).
- `variogram.png` — empirical semivariogram of the station values.
- `before_after_bars.png` — grouped RMSE/MAE bars (kriging vs ML).
- `summary.md` — human-readable table.

### Tests (fast, CPU)

```bash
python -m pytest experiments/06_spatial_interpolation/tests common/tests/test_spatial.py -q
```

CPU-only and under 30 s. The generator tests assert the field is deterministic,
well-shaped, and that the **covariate is genuinely informative** (and varies on
shorter scales than the GRF). The experiment tests assert IDW/kriging/ML all return
finite metrics, the variogram rises with lag, kriging returns an uncertainty surface,
and the headline: **RMSE_ml ≤ RMSE_kriging** on a held-out grid (and kriging < IDW).

---

## Swapping in real data

The generator returns a `SpatialDataset` with train arrays
(`train_coords[n,2]`, `train_covariate[n]`, `train_value[n]`) and a dense grid
(`grid_coords[m,2]`, `grid_covariate[m]`, `grid_value[m]`, `grid_shape`). **Keep
those names/shapes and every script here works unchanged.** To use real data:

1. **Stations** — observed 2 m temperature from **GHCN-Daily** or a national met
   service (or a pollutant from **OpenAQ**): build `train_coords` (lon/lat or, better,
   a projected CRS in km), `train_value`.
2. **Covariate** — sample **elevation** from a **DEM** (SRTM or the Copernicus DEM) at
   each station (`train_covariate`) and resample the DEM to your target prediction
   grid (`grid_covariate`).
3. **Scoring grid** — `grid_coords` over the target raster; `grid_value` from a
   held-out station subset (or a trusted reanalysis) for honest evaluation.

Feed it straight into `run_before` / `run_after`. (Standardise coords to km and scale
the covariate as the generator does.) For the *combined* hero method, plug the
covariate into a regression-kriging / GP-mean step.

---

## References (described generically — no invented IDs/stats)

- **Kriging / variograms** — the foundations of geostatistics: Matheron's regionalized
  variables, the semivariogram, ordinary/universal kriging (textbook treatments:
  Cressie, *Statistics for Spatial Data*; Chilès & Delfiner, *Geostatistics*; Isaaks
  & Srivastava, *Applied Geostatistics*; Goovaerts, *Geostatistics for Natural
  Resources Evaluation*).
- **GP ≈ kriging** — a Gaussian process with a stationary covariance kernel is the
  Bayesian formulation of kriging; the RBF kernel corresponds to a Gaussian
  variogram, its length scale to the range, and the WhiteKernel to the nugget
  (Rasmussen & Williams, *Gaussian Processes for Machine Learning*).
- **Regression-kriging / kriging with external drift** — the standard way to fold a
  covariate (e.g. a DEM) into kriging to get both lower error *and* an uncertainty
  surface (Hengl and colleagues; the broader environmetrics literature).
- **Random forests / gradient boosting for spatial prediction** — tree ensembles with
  spatial and covariate features are now common for environmental mapping; they
  trade kriging's calibrated variance for flexible covariate response.
- **Datasets for a real swap** — GHCN-Daily / OpenAQ (observations); SRTM / Copernicus
  DEM (elevation covariate).

No statistics, dates, or specific results are invented here; the headline numbers
live in the committed `results/` from an actual run on this machine.
