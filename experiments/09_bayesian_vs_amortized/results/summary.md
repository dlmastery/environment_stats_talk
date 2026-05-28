# Experiment 09 — Bayesian hierarchical vs amortized inference: BEFORE vs AFTER

- Device: `cuda`  |  stations S = 30  |  generative (mu, tau, sigma) = (0.5, 1.2, 0.7)
- MH iters: 4000 (burn 1500)   |  Amortized training datasets: 4000   |  Quick: False

## Per-station posterior quality vs the TRUE theta_s

| Method | Coverage (95%) ~ 0.95 | RMSE-to-truth ↓ | Mean interval width ↓ |
|--------|------:|------:|------:|
| Before: closed-form (known hyper) | 0.967 | 0.153 | 0.720 |
| Before: MCMC (MH, full posterior) | 1.000 | 0.152 | 0.751 |
| **After: amortized (MLP)** | **1.000** | **0.266** | **2.349** |

Coverage is the fraction of stations whose true effect falls inside the reported 95% CI; both Bayesian methods aim for ~0.95 (nominal). RMSE-to-truth summarizes how close the *posterior means* are to the latent truth (lower is sharper for the same coverage).

## MCMC posterior over hyperparameters (informative diagnostics)

- mu ~ 0.326  (95% CI -0.048..0.717, true 0.5)
- tau ~ 1.036  (95% CI 0.781..1.379, true 1.2)
- sigma ~ 0.735  (95% CI 0.689..0.785, true 0.7)
- MH acceptance rates: {'theta': 0.5284083333333334, 'mu': 0.62625, 'tau': 0.51625, 'sigma': 0.2655}

## Wall-time honest accounting (this machine)

| Step | Time (s) |
|------|---------:|
| BEFORE closed-form (known hyper) | 0.0000 |
| BEFORE MCMC (MH) fit             | 11.6112 |
| AFTER amortized TRAIN (one-time) | 24.4593 |
| AFTER amortized SCORE (per new dataset) | 0.0080 |

**Speedup of amortized SCORING over MH FIT on a fresh dataset: ~1451.6x.** Honest caveat: that one-time training cost is amortized over *all* future datasets from the same model family; if the model class changes you must retrain.

Artifacts: `posterior_intervals.png`, `before_after_bars.png`, `metrics.json`.
