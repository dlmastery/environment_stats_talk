# RESULTS — the before/after scoreboard

**Every number below was produced by the code in this repo on this machine** (Windows, Python 3.12, NVIDIA RTX 4090 Laptop GPU). Synthetic + small data, deterministic seeds → reproducible with one command per experiment. Reproduce all at once with `python run_all_tests.py` (tests) and each experiment's `run_before_after.py` (results).

> **Read this honestly.** "AFTER" means *Claude Code + an AI-for-science model wrote, ran, and benchmarked the workflow*. The headline is the collapse in **human effort** (days of bespoke coding → one command in minutes) and the **breadth of rigor** added — *not* a claim that a neural net always beats classical statistics. Across the **7 experiments** below the verdicts are mixed *on purpose*: clean AFTER wins where the structure rewards it (hydrology +0.56 NSE, conformal calibration ~7× tighter, hard-mode land cover +0.36 acc, biodiversity recall), a **rigor/uncertainty** win (extremes), and honest **ties / classical-wins** where they belong (1-step forecasting: **SARIMA beats the LSTM**; easy land cover: RF = CNN). That mix *is* the point — the agent lets you find out fast, fairly, and reproducibly.

---

## Scoreboard

| # | Task | BEFORE (traditional) | AFTER (Claude Code + AI-for-science) | What AFTER adds | Honest verdict |
|---|------|----------------------|--------------------------------------|-----------------|----------------|
| 01 | Daily temperature forecasting (20 yr, RTX 4090) | persistence RMSE **4.31 °C** at 14-day lead | LSTM RMSE **2.94** (skill **+0.318**); SARIMA RMSE **2.10** | persistence + seasonal-naive + SARIMA + LSTM **all built & skill-scored in one run** | **SARIMA wins at ≥7-day lead** — classical beats the net here; the win is *fair, fast comparison* |
| 02 | Precipitation extremes & trend (40 yr) | empirical 100-yr return level **48.6 mm** (point estimate, no CI, no checks) | GEV 100-yr **55.8 mm [46.7–80.2]** + Mann-Kendall trend **p=0.0011** | bootstrap CIs, multi-index trends, **a `validate()` gate that flags its own stationarity violation** | **Rigor win** — same question, far more honest uncertainty; empirical method was biased low |
| 03 | Species interactions from citizen text | keyword/regex: precision **0.67**, recall **0.10**, F1 **0.18** | structured extraction: P/R/F1 **1.00** on the closed synthetic corpus | passive-voice & synonym handling → an interaction network | **Recall 10% → 100%** (synthetic ceiling — *not* a real-world number; real data needs human verification) |
| 04 | Land cover + change (RTX 4090) | RF on NDVI/NDWI indices | CNN on raw 5-band cube + change map | learns texture indices average away | **Two regimes, both honest:** easy classes → RF = CNN = **1.00** (tie, effort/capability shift); **hard mode** (texture-only classes) → RF **0.642** vs CNN **0.998** (**+0.356 acc**, a real win) |
| 05 | **Autonomous study (autoresearch loop)** | a statistician hand-runs ~6 experiments over days | Claude Code ran a **gated 6-experiment loop** in ~38 s; champion **+8.0% skill** (RMSE 1.95→1.79 °C), monotone composite | citation gate + numeric-prediction gate + Goodhart-proof frozen composite + crash-recovery checkpoint | **Process win** — gates block a shallow guess (proven by test); the agent even *rejected* an obvious idea (more lags hurt a near-AR(1) series) |
| 08 | Hydrology: rainfall-runoff streamflow (RTX 4090) | linear / conceptual bucket: NSE **0.14** | LSTM rainfall-runoff: NSE **0.70** | carries catchment state (soil moisture, snow, routing) the linear model can't | **Clean AFTER win** — **+0.56 NSE / +0.51 KGE**; the classic LSTM-hydrology result, gap widens with data |
| 12 | Uncertainty: prediction intervals | normal-theory PIs: 80% nominal → **86.9%** empirical (miscalibrated under heavy tails) | split/normalized **conformal** PIs | distribution-free finite-sample marginal coverage | **Calibration + sharpness win** — mean gap **0.033 → 0.004 (~7×)** *and* narrower bands (6.37→5.22 °C at 80%) |

Artifacts for each row live in `experiments/NN_*/results/` (metrics JSON, plots, `summary.md`).

---

## The real "speedup" (labor, not just wall-clock)

The transformation is in **person-time**, which is what the talk's before/after slides quantify:

| Task | BEFORE person-effort (typical) | AFTER wall-time (this repo) |
|------|-------------------------------|-----------------------------|
| Build + benchmark 4 forecasters with skill scores + plots | ~a day of coding | one `run_before_after.py` (minutes) |
| GEV + bootstrap CIs + multi-index trend + assumption checks | ~a day | one command (≈ a minute) |
| Hand-label interactions in hundreds of comments | hours–days | one extraction pass (seconds) |
| Train + evaluate a CNN classifier + change map | hours (setup + tuning) | one command on the 4090 |
| Run a disciplined 6-experiment model search | days, easy to fool yourself | the gated loop (seconds), Goodhart-proof |

This is the "science 100× faster" claim stated precisely: **the agent compresses the engineering and bookkeeping, and adds rigor gates a rushed human often skips** — while the human keeps every scientific decision (below).

## What AFTER does **not** change

- The human owns the **research question**, the **assumptions**, **model choice**, **multiple-testing** discipline, **uncertainty** interpretation, and **peer review**.
- AI **proposes and benchmarks**; the human **decides and signs off** (see `skills/env-stats-validate`).
- Synthetic-data numbers are **ceilings/illustrations**, not real-world performance. Each experiment documents the real-data swap (ERA5/GBIF/Sentinel-2/…).
- Speculative or future-dated claims are quarantined in [`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md) and never reported as fact.
