# Glossary — a statistics ↔ machine-learning bridge

The talk's audience is mixed: environmental statisticians who know GEV, kriging, and
Mann-Kendall cold, and may be newer to LSTMs and conformal prediction; and ML-leaning
attendees for whom the reverse is true. This glossary defines, in 1–3 plain sentences,
every term that recurs in the experiments and [`RESULTS.md`](../RESULTS.md). Definitions
are standard and textbook; where a precise attribution would be needed for a slide,
confirm it against the primary source — **(verify)**.

## Error & accuracy metrics

- **RMSE (root-mean-square error).** The square root of the average squared
  prediction error, in the same units as the target. Lower is better; it penalises large
  errors more than small ones, so it is sensitive to outliers/spikes.

- **MAE (mean absolute error).** The average absolute prediction error, in the target's
  units. Lower is better; more robust to outliers than RMSE because errors are not squared.

- **ACC (anomaly correlation coefficient).** The correlation between predicted and
  observed *anomalies* (departures from a climatology), widely used to score weather/climate
  forecasts. Ranges to 1 (perfect pattern match); an operational rule of thumb treats
  ACC > 0.6 as useful skill **(verify the exact threshold convention you cite)**.

- **Skill score.** A model's error expressed relative to a reference baseline:
  `skill = 1 − error_model / error_reference`. 1 = perfect, 0 = no better than the
  reference, < 0 = worse. "Low RMSE" alone is meaningless without naming the reference.

- **NSE (Nash–Sutcliffe efficiency).** The standard streamflow goodness-of-fit: 1 minus
  the ratio of model error variance to the variance of the observations. 1 = perfect,
  0 = no better than predicting the long-term mean flow, < 0 = worse than the mean. **(verify
  Nash & Sutcliffe 1970.)**

- **KGE (Kling–Gupta efficiency).** A streamflow score that decomposes performance into
  three parts — correlation, bias ratio, and variability ratio — so you can see *why* a
  hydrograph is off, not just that it is. 1 = perfect; higher is better. **(verify Gupta et
  al. 2009.)**

- **Composite metric.** A single aggregate score the autoresearch loop optimises, combining
  validation/test performance with a penalty for folds that fall below a threshold (default:
  `min(val, test) − penalty · n_below_threshold`). It is **frozen** (fingerprinted) at the
  start so it cannot be quietly redefined to flatter a result. See *Goodhart*.

## Extremes & trends

- **GEV (generalized extreme value distribution).** The limiting distribution of block
  maxima (e.g. annual maximum daily rainfall); its three parameters (location, scale, shape)
  are fit by maximum likelihood and used to extrapolate rare events. **(verify Coles, *An
  Introduction to Statistical Modeling of Extreme Values*.)**

- **POT (peaks-over-threshold).** An alternative extremes approach that models all
  exceedances above a high threshold (typically with a generalized Pareto distribution),
  using more of the tail data than block maxima. **(verify.)**

- **Return level.** The value an extreme is expected to exceed once per *return period*
  (e.g. the "100-year flood" is the level exceeded on average once in 100 years). It is an
  extrapolation from a fitted tail and **must carry a confidence interval**, which widens
  with the return period.

- **Mann–Kendall test.** A non-parametric, rank-based test for a monotonic trend in a time
  series; it makes no distributional assumption but **assumes independence**, so
  autocorrelation must be addressed (pre-whitening / block bootstrap) before trusting its
  p-value. **(verify Mann; Kendall.)**

- **Sen's slope (Theil–Sen estimator).** A robust trend-magnitude estimator: the median of
  all pairwise slopes between data points. Pairs naturally with Mann-Kendall (which tests
  *whether* there is a trend; Sen's slope estimates *how steep*). **(verify Theil; Sen.)**

## Spatial / geostatistics

- **Kriging.** The geostatistical interpolator that predicts a value at an unobserved
  location as a weighted average of nearby observations, with weights derived from the
  spatial autocorrelation model — and, crucially, returns a **predictive-variance surface**
  (uncertainty that grows away from data). Ordinary kriging is the common variant. A Gaussian
  process with a stationary covariance kernel is the Bayesian formulation of kriging.

- **Variogram (semivariogram).** A diagnostic plotting how dissimilar two observations are
  as a function of the distance between them; it rises with separation toward a *sill*,
  confirming there is exploitable spatial autocorrelation and revealing the *range* and
  *nugget*. **(verify Matheron.)**

- **IDW (inverse-distance weighting).** The simplest spatial interpolator: predict a weighted
  average where weights fall off as `1/distance^p`. Transparent and assumption-light, but it
  has no statistical model and gives **no uncertainty estimate** — usually a weak baseline.

## Uncertainty & calibration

- **Conformal prediction.** A distribution-free wrapper that turns any point predictor into
  prediction intervals by reading the band width off the empirical quantile of held-out
  calibration residuals. For exchangeable data it gives **finite-sample marginal coverage**
  guarantees with no Gaussianity assumption.

- **Marginal coverage.** The guarantee that, *averaged over the input distribution*, a
  prediction interval contains the true value at least as often as its nominal level (e.g. a
  90% interval covers ≥ 90% of cases on average). It does **not** guarantee *conditional*
  coverage (the right rate in every sub-region), which is impossible distribution-free in
  general.

## Validation & splitting

- **Walk-forward (rolling) validation.** A time-series-correct evaluation where train/
  validation/test windows roll forward in time, so the test set is always strictly in the
  *future* of training — with a *purge + embargo* around the boundary sized to the label
  horizon. Never shuffle a time series.

- **GroupKFold (spatial group split).** Cross-validation that keeps whole groups — basins,
  grid cells, stations, regions — entirely on one side of each split, so autocorrelated
  neighbours never straddle train and test. A random point/pixel split leaks because nearby
  points are nearly identical.

- **Super-fold.** A robustness split that unions multiple *regime* windows (e.g. by climate
  regime, ENSO phase, or wet-vs-dry years) and judges the model on the **worst regime**, not
  just the average — the environmental analogue of a stress test.

- **Persistence baseline.** The "tomorrow ≈ today" (random-walk) forecast; the canonical
  reference for forecast *skill* because highly autocorrelated series (like daily
  temperature) make it genuinely hard to beat at short horizons. If a model can't beat
  persistence, it has no skill.

## ML models & failure modes

- **LSTM (long short-term memory network).** A recurrent neural network that carries an
  internal *cell state* across a sequence, letting it learn long-range, state-dependent
  memory — e.g. antecedent soil moisture and snowpack in rainfall-runoff. **(verify
  Hochreiter & Schmidhuber 1997.)**

- **Foundation model.** A large model pre-trained on broad data that can be applied to new
  tasks with little or no task-specific training — e.g. time-series foundation models
  (TimesFM, Chronos, MOMENT, Moirai) used for *zero-shot* forecasting (forecasting a new
  series with no fitting). **(verify each model's release.)**

- **Goodhart('s law).** "When a measure becomes a target, it ceases to be a good measure."
  In practice: an optimiser can make a score look better by quietly redefining the score
  rather than improving the work — which is why the loop *freezes and fingerprints* its
  composite metric so a mid-project rewrite raises an error.
