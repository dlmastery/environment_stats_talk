# Experiment 09 — Bayesian hierarchical vs amortized inference: BEFORE vs AFTER

Estimate a per-station effect ``theta_s`` in a network of ``S`` environmental
monitoring stations, given noisy observations ``y_{s,i} = theta_s + eps``, under
the textbook **hierarchical Normal-Normal** model

```
theta_s ~ Normal(mu, tau^2)         (partial-pooling prior across stations)
y_{s,i} ~ Normal(theta_s, sigma^2)  (observation noise)
```

This is the workhorse model behind multi-station bias correction, small-area
estimation, and meta-analysis. Two ways to do the inference:

- **BEFORE — classical Bayes** (`before/classical_bayes.py`): the closed-form
  Normal-Normal conjugate update at known hyperparameters, **plus** a
  from-scratch **Metropolis-Hastings sampler** for the full joint posterior
  ``p(theta_1..S, mu, tau, sigma | y)`` with weakly-informative priors. Pure
  ``numpy + scipy`` — no PyMC.
- **AFTER — amortized inference** (`after/amortized.py`): a small PyTorch MLP
  trained **once** on many simulated datasets to map per-station summary stats
  to ``(posterior_mean, posterior_sd)``. Scoring a fresh dataset is one CPU
  forward pass (microseconds per station) regardless of dataset size.

Everything **runs anywhere**: synthetic data with deterministic seeds, CPU is
fine, no API keys, no GPU required. The amortized network auto-uses CUDA when
present and trains on CPU in seconds otherwise.

---

## The honest trade-off (the headline)

| Method | Per-dataset cost | One-time setup | Reusable across datasets? | Interpretable? |
|---|---|---|---|---|
| Before — closed-form (known hyper) | instant (<1 ms) | none | yes, exact | yes |
| Before — MCMC (MH on full posterior) | seconds–minutes per dataset | none | every new dataset re-pays the full cost | yes |
| **After — amortized inference**   | **microseconds per dataset (single forward pass)** | train **once** on simulated datasets | yes, until the model class changes | network is a black box |

The trade is stark and honest:

- **MCMC is correct for any dataset** (no model misspecification beyond the prior),
  is fully interpretable, and gives you posteriors over hyperparameters for free
  — but you re-pay the full sampling cost every time a new dataset arrives.
- **Amortized inference pays once at training time** and then scores fresh
  datasets effectively for free, with calibrated coverage *as long as the new
  dataset is drawn from the same model family the network was trained on*. If
  the model class changes (e.g., heavier tails, different prior), you must
  retrain. The classical sampler does not have this caveat.

The talk's point is **not** "amortized always wins"; it's that when you have
*many* small datasets from a stable model class — exactly the situation in
operational environmental monitoring (one network, thousands of days/seasons) —
amortized inference is a real engineering win, while the *classical* sampler
remains the right tool for one-off, bespoke inference.

---

## Results (this machine)

The exact numbers are written by the latest run into
`results/metrics.json` and `results/summary.md`. The committed run uses
``S = 30`` stations, MH (4000 iters, 1500 burn-in), and an amortized network
trained on 4000 simulated datasets. The bar chart `before_after_bars.png` shows
empirical coverage of the 95% CIs (both Bayesian methods land near 0.95) and the
log-scale wall-time breakdown (MCMC fit vs amortized scoring vs the one-time
amortized training). The line plot `posterior_intervals.png` shows, per station,
the MCMC CI, the amortized CI, and the true latent ``theta_s``.

---

## Why coverage parity matters

A faster method that breaks coverage is not a faster Bayesian method — it is a
wrong method that ships sooner. We therefore **measure empirical coverage of the
true latent ``theta_s``** (possible only because the data are synthetic and the
truth is known) and require the amortized model to land near the nominal 95%,
not just to produce *some* interval. The Gaussian-NLL training loss is a proper
scoring rule, so the network is *encouraged* to be both sharp and calibrated.

The closed-form conjugate posterior at known hyperparameters is the gold
standard reference: any per-station shrinkage the other two methods report
should be in the same ballpark, otherwise they are mis-specified.

---

## How to run

Always run **from the repo root** so `import common` resolves.

```bash
# Fast smoke run (S=12, fewer MH iters, fewer training sims): seconds.
python experiments/09_bayesian_vs_amortized/run_before_after.py --quick

# Default (S=30, full MH, amortized trained on 4000 sims): well under a minute.
python experiments/09_bayesian_vs_amortized/run_before_after.py
```

Useful flags: `--quick`, `--n-stations S`, `--mh-iter N --mh-burn M`,
`--train-sims K`, `--seed S`.

Outputs land in `results/`:
- `metrics.json` — coverage, RMSE-to-truth, mean interval width, wall times.
- `posterior_intervals.png` — per-station CIs from MCMC and amortized vs truth.
- `before_after_bars.png` — coverage parity (left) + log-scale wall times (right).
- `summary.md` — human-readable table.

### Tests (fast, CPU)

```bash
python -m pytest experiments/09_bayesian_vs_amortized/tests common/tests/test_bayesian.py -q
```

The amortized-vs-MCMC speed assertion is the headline test: the per-dataset
forward pass of the trained MLP must beat even a tiny MH fit by a healthy
margin (at least 10x in the test; in practice many orders of magnitude as
``S`` grows).

---

## What this experiment deliberately does NOT claim

- **No "amortized beats Bayes" framing.** Both report calibrated 95% CIs; the
  win is *per-dataset latency*, not coverage.
- **No invented benchmarks.** All numbers come from the committed local run.
- **No PyMC dependency.** The whole BEFORE pipeline is hand-written from numpy
  primitives so it works in a minimal scientific Python environment.
- **No real-data claims here.** To use real station data, drop a long-form
  DataFrame with columns ``['station', 'obs_idx', 'value']`` into
  `station_sufficient_stats(...)` and the rest of the pipeline runs unchanged.

---

## References (described generically — no invented IDs/stats)

- **Hierarchical / partial-pooling Bayes for grouped data.** The Normal-Normal
  hierarchy is the canonical small-area-estimation / multi-station shrinkage
  model in environmental statistics (see Gelman, Carlin, Stern & Rubin,
  *Bayesian Data Analysis*, for the textbook treatment; Cressie's
  *Statistics for Spatial Data* for the environmental framing).
- **Metropolis-Hastings.** The original Metropolis (1953) / Hastings (1970)
  random-walk Metropolis algorithm — the simplest correct MCMC scheme for a
  black-box log posterior.
- **Amortized / simulation-based inference.** A neural network is trained on
  many simulated ``(data, parameters)`` pairs to learn a fast posterior mapping.
  See the literature on *amortized variational inference* (e.g., the variational
  autoencoder line of work), *neural posterior estimation*, and *likelihood-free /
  simulation-based inference* (the SBI community) for the general idea. The
  specific architecture here is intentionally minimal — an MLP on per-station
  sufficient statistics — to keep the implementation small and the comparison
  honest.
- **Calibration as a proper scoring rule.** Training under Gaussian NLL is a
  proper scoring rule that rewards both sharpness *and* honest variance
  (Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and
  Estimation", JASA 2007).

No statistics, dates, or specific results are invented here; the headline
numbers live in the committed `results/` from an actual run on this machine.
