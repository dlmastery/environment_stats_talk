# Experiment 12 — Uncertainty quantification: conformal prediction intervals (BEFORE vs AFTER)

A result with no uncertainty is not yet a scientific result — and an interval that
*claims* 90% coverage but only delivers 87% (or wastes width at 93%) is worse than
no interval, because it is trusted. This experiment is the talk's **rigor showcase**
for uncertainty: build one transparent point forecaster for daily temperature, then
put a prediction interval around it two ways and **check the empirical coverage**.

- **BEFORE** (`before/normal_theory.py`): textbook Gaussian / normal-theory bands,
  `yhat ± z·sigma_hat`, where `sigma_hat` is the training-residual standard
  deviation. Correct *if* residuals are i.i.d. Normal with constant variance.
- **AFTER** (`after/conformal.py`): **split (inductive) conformal** intervals read
  off the empirical quantile of held-out calibration residuals — plus a
  **locally-adaptive (normalized) conformal** variant whose width adapts to the
  local residual spread. Distribution-free, finite-sample **marginal coverage**.

Everything **runs anywhere**: synthetic data (deterministic seeds), CPU-only, no
torch, no GPU, no API keys. The point forecaster is a ridge regression on lag
features — deliberately light, because the *intervals* are the subject, not the
regressor (the same point predictions feed both methods, so the comparison is fair).

---

## The story

**Why the normal-theory band fails here (honest, not rigged).** The synthetic
series has seasonal **heat extremes** (a heavy right tail) that the warming **trend
intensifies over time** — these are real knobs of `common.synthetic_climate`
(`heat_extreme_rate`, `warming_c_per_decade`), set to a realistic heteroscedastic
regime, not a contrived distribution. The forecast residuals are therefore
**heteroscedastic** (wider in the warm season) and **right-skewed with heavy tails**
(measured excess kurtosis ≈ 5–7). A *single* symmetric Gaussian `sigma_hat`
mis-sizes the band: the heavy tail inflates the standard deviation, so the band
**over-covers in the body** (it is too wide for typical days) while mis-handling the
tail. Empirical coverage drifts off nominal, and the intervals are needlessly wide.

**Why conformal fixes it.** Split conformal scores each calibration point by its
absolute residual and reads the band half-width off the
`(1−α)(1 + 1/n_cal)`-empirical quantile of those scores. Because calibration and
test scores are exchangeable, a fresh point is covered with probability **≥ the
nominal level** — *with no distributional assumption*. The band is sized to the
*actual* residual distribution (skew and tails included), so it lands on nominal
with **smaller** width. The **normalized** variant divides the score by a learned
conditional-spread estimate `sigma_hat(x)` and multiplies the band back by it,
giving **variable width** that widens in the warm season and tightens in the calm
season — targeting conditional coverage the constant-width band cannot.

---

## Results (committed full run: 20 years, seed 0, heteroscedastic regime)

Empirical **test** coverage and mean interval width at three nominal levels. The
calibration gap `|empirical − nominal|` is what we want near zero.

| Nominal | Method | Empirical coverage | \|gap\| | Mean width (°C) |
|--------:|--------|-------------------:|-----:|----------------:|
| 80% | before: normal-theory | 0.869 | 0.069 | 6.37 |
| 80% | after: split conformal | **0.791** | **0.009** | **5.22** |
| 80% | after: normalized conformal | **0.813** | **0.013** | 5.49 |
| 90% | before: normal-theory | 0.929 | 0.029 | 8.18 |
| 90% | after: split conformal | **0.904** | **0.004** | **7.27** |
| 90% | after: normalized conformal | **0.897** | **0.003** | 7.36 |
| 95% | before: normal-theory | 0.950 | 0.000 | 9.75 |
| 95% | after: split conformal | **0.950** | **0.000** | 9.72 |
| 95% | after: normalized conformal | **0.943** | 0.007 | 9.26 |

**Mean absolute calibration gap, averaged over levels (lower is better):**

| Method | Mean \|empirical − nominal\| |
|--------|-----------------------------:|
| before: normal-theory | **0.0329** |
| after: split conformal | **0.0044** |
| after: normalized conformal | **0.0077** |

**The honest win.** Conformal shrinks the average calibration gap from **0.033 to
0.004** (split) — roughly a **7× reduction** — *and* does it with **narrower**
intervals (e.g. 80% width 6.37 → 5.22 °C). At 80% the normal-theory band over-covers
badly (0.869 vs 0.80) because the heavy-tailed `sigma_hat` is inflated; at 95% the
two agree because the tail matters less there. See `results/coverage_plot.png`
(distance from the `y = x` diagonal = miscalibration) and `results/interval_plot.png`
(the conformal band is visibly tighter and the normalized band breathes with the
season).

> Numbers above are reproduced exactly by `run_before_after.py` (no `--quick`) and
> are committed in `results/metrics.json` / `results/summary.md`. `--quick` (6 years)
> tells the same story with a slightly larger gap.

---

## What conformal guarantees — and what it does NOT

**Guarantees (the part you can put on a slide):**

- **Distribution-free marginal coverage.** For exchangeable data and *any* point
  predictor, the split-conformal interval covers a fresh target with probability
  **≥ the nominal level**, in **finite samples** — no Gaussianity, no
  homoscedasticity, no asymptotics. The `+1/n` term in the quantile rank is the
  exact finite-sample correction.
- **Model-agnostic.** Wrap the same recipe around ridge, an LSTM, a gradient-boosted
  tree, or a foundation model — only the residual scores change.

**Does NOT guarantee (the caveats an environmental statistician must state):**

- **Conditional coverage.** The guarantee is *marginal* (averaged over the input
  distribution). A constant-width band can still under-cover the volatile warm
  season and over-cover the calm season even while hitting the level on average. The
  **normalized** variant *improves* conditional coverage by letting width vary, but
  does **not** guarantee it; exact conditional coverage is impossible distribution-
  free in general.
- **Exchangeability.** Forecast residuals are weakly **serially dependent**, which
  breaks strict exchangeability. We mitigate this with a calibration block adjacent
  to the test block (so the calibration residual distribution resembles the test
  one) and we **verify empirical coverage on a held-out test set** rather than
  trusting the guarantee blindly. For strongly non-exchangeable / shifting series,
  use a time-series-aware conformal scheme (e.g. weighted / adaptive conformal that
  updates the quantile online); the split recipe here is the transparent baseline.
- **Calibration-set size.** Very high levels need enough calibration points: if
  `(n_cal + 1)·level > n_cal`, the honest answer is an infinite band — the code
  returns `+inf` rather than a falsely finite interval.

This is exactly the `env-stats-validate` check #5 ("calibrated uncertainty, with a
coverage check"): **AI proposes the interval; the human verifies its coverage** and
states what it does and does not promise.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (6 years; same story, slightly larger gap): a few seconds on CPU
python experiments/12_conformal_uncertainty/run_before_after.py --quick

# Committed full run (20 years, levels 80/90/95): a few seconds on CPU
python experiments/12_conformal_uncertainty/run_before_after.py

# Custom levels / data length
python experiments/12_conformal_uncertainty/run_before_after.py --levels 0.8 0.9 0.95 0.99 --n-years 30
```

Useful flags: `--quick`, `--n-years N`, `--levels L1 L2 ...`, `--lookback K`,
`--alpha A` (ridge strength), `--seed S`.

Outputs land in `results/`:

- `metrics.json` — nominal vs empirical coverage + mean width for all three methods
  at every level, plus the mean calibration gap and config/provenance.
- `coverage_plot.png` — nominal vs empirical coverage; the dotted `y = x` line is
  perfect calibration.
- `interval_plot.png` — test-tail observations with the normal-theory and conformal
  bands overlaid (90% level).
- `summary.md` — the human-readable table.

### Tests (fast, CPU, < 30 s)

```bash
python -m pytest experiments/12_conformal_uncertainty/tests -q
```

The suite checks: conformal empirical coverage within tolerance of nominal;
conformal's average calibration gap strictly smaller than normal-theory's in the
heteroscedastic regime; the finite-sample quantile correction; normalized-band width
increasing with the local spread estimate; and that all widths are positive/finite.
Test logic is sub-second (the wall time is one-time `sklearn`/`scipy` import).

---

## Swapping in real data

The point forecaster consumes any daily series with the project's schema — a
DataFrame indexed by `date` with a single value column (here `t2m`). To use **real
ERA5 2 m temperature** (or a station record), load it into that schema and the whole
pipeline runs unchanged:

```python
import xarray as xr
ds = xr.open_dataset("era5_t2m.nc")
t2m = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")
daily = (t2m.resample(time="1D").mean() - 273.15)         # K -> °C
series = daily.to_dataframe(name="t2m")["t2m"]
```

Then split chronologically into **train / calibration / test** (never shuffle a time
series — see `env-stats-validate` #1), and call
`before.normal_theory.run_before(train, test)` and
`after.conformal.run_after(train, cal, test)`. Conformal's only requirement is a
held-out calibration set the model never trained on; everything else — the heavy
precipitation tails, the non-Gaussian station residuals — is exactly where the
distribution-free guarantee earns its keep. For strongly non-stationary real series,
prefer a time-series-aware conformal variant (weighted/adaptive) as noted above.

---

## Method notes (generic, no invented citations)

Split / inductive conformal prediction and its locally-adaptive (normalized)
variant are standard, well-established tools in the conformal-prediction and
distribution-free-inference literature; this experiment implements the textbook
split recipe directly from first principles (calibration-residual quantile with the
finite-sample `+1/n` correction). No specific arXiv identifiers, authors, dates, or
benchmark statistics are asserted here — per the project's zero-hallucination rule,
only the verifiable mechanics and the experiment's own measured numbers are claimed.
