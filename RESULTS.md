# RESULTS — the before/after scoreboard

**Every number below was produced by the code in this repo on this machine** (Windows, Python 3.12, NVIDIA RTX 4090 Laptop GPU). Synthetic + small data, deterministic seeds → reproducible with one command per experiment. Reproduce all at once with `python run_all_tests.py` (tests) and each experiment's `run_before_after.py` (results).

> **Read this honestly.** "AFTER" means *Claude Code + an AI-for-science model wrote, ran, and benchmarked the workflow*. The headline is the collapse in **human effort** (days of bespoke coding → one command in minutes) and the **breadth of rigor** added — *not* a claim that a neural net always beats classical statistics. In two of five cases the honest result is "the classical method wins / ties," and we keep it visible. That is the point: the agent lets you find out fast.

---

## Scoreboard

| # | Task | BEFORE (traditional) | AFTER (Claude Code + AI-for-science) | What AFTER adds | Honest verdict |
|---|------|----------------------|--------------------------------------|-----------------|----------------|
| 01 | Daily temperature forecasting (20 yr, RTX 4090) | persistence RMSE **4.31 °C** at 14-day lead | LSTM RMSE **2.94** (skill **+0.318**); SARIMA RMSE **2.10** | persistence + seasonal-naive + SARIMA + LSTM **all built & skill-scored in one run** | **SARIMA wins at ≥7-day lead** — classical beats the net here; the win is *fair, fast comparison* |
| 02 | Precipitation extremes & trend (40 yr) | empirical 100-yr return level **48.6 mm** (point estimate, no CI, no checks) | GEV 100-yr **55.8 mm [46.7–80.2]** + Mann-Kendall trend **p=0.0011** | bootstrap CIs, multi-index trends, **a `validate()` gate that flags its own stationarity violation** | **Rigor win** — same question, far more honest uncertainty; empirical method was biased low |
| 03 | Species interactions from citizen text | keyword/regex: precision **0.67**, recall **0.10**, F1 **0.18** | structured extraction: P/R/F1 **1.00** on the closed synthetic corpus | passive-voice & synonym handling → an interaction network | **Recall 10% → 100%** (synthetic ceiling — *not* a real-world number; real data needs human verification) |
| 04 | Land cover + change (RTX 4090) | Random Forest on NDVI/NDWI indices: acc **1.00** | CNN on raw 5-band cube: acc **1.00**; **change map** at F1 **1.00** | learns texture; **change detection is an AFTER-only capability** | **Tie on accuracy** (synthetic classes too separable) — honest effort/capability shift, not an accuracy win |
| 05 | **Autonomous study (autoresearch loop)** | a statistician hand-runs ~6 experiments over days | Claude Code runs the **gated 7-step loop**, tracking a champion | citation gate + numeric-prediction gate + Goodhart-proof composite + crash-recovery | _pending — flagship loop running; see `experiments/05_autoresearch_climate/`_ |

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
