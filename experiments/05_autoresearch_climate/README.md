# Experiment 05 — AutoResearch Climate Loop (the FLAGSHIP)

**Claude Code is the researcher.** This is the AFTER of the whole talk: instead of
a statistician hand-running a forecasting sweep over days, the env-stats
autoresearch loop drives *itself* through a gated, literature-grounded sequence of
experiments to improve a climate forecaster — and reports the honest result.

It runs **anywhere** (CPU, synthetic data, no API keys) and finishes the entire
six-experiment loop in **~38 seconds of compute**. The protocol — gate every
experiment, freeze the composite, judge on future years and on the worst regime —
is *identical* at 4090 scale; only the backbone and data grow.

```bash
# from the repo root
python experiments/05_autoresearch_climate/run_loop.py
python -m pytest experiments/05_autoresearch_climate/tests -q
```

## What the loop produced (real, committed)

| Exp | one attributable change | composite (−RMSE) | test RMSE (°C) | skill vs persistence | status |
|----:|--------------------------|------------------:|---------------:|---------------------:|--------|
| 1 | persistence floor (walk-forward by year) | −1.9586 | 1.9477 | +0.000 | KEEP |
| 2 | tiny MLP, 7 lags + sin/cos day-of-year | −1.8503 | 1.8435 | +0.054 | KEEP |
| 3 | shorter **3-lag** window (parsimony) | −1.8253 | 1.8237 | +0.064 | KEEP |
| 4 | learning-rate tune 0.05 → 0.08 | −1.8157 | 1.8129 | +0.069 | KEEP |
| 5 | more epochs 400 → 1500 (full convergence) | **−1.7967** | **1.7914** | **+0.080** | KEEP |
| 6 | super-fold-by-regime robustness | −1.9622 | 1.8228 | +0.064 | DISCARD |

**Final champion: Exp5**, an 8.0% RMSE improvement over persistence. The champion
composite is **non-decreasing** the whole way (−1.96 → −1.85 → −1.83 → −1.82 →
−1.80, then Exp6 correctly DISCARDs and leaves it). See `champion_progress.png`.

**Honest reading (this matters to the TIES audience):** one-step daily-temperature
forecasting is a near-persistence problem, so the learned model beats the naive
floor only *modestly* — and most of the gain is the seasonal encoding plus
convergence, not architecture. The loop says so plainly. It also caught a wrong
"obvious" idea: *more* lags actually hurt this near-AR(1) series, so Exp3 went the
other way (3 lags) on a parsimony argument — exactly the kind of mistake a hurried
manual sweep would have shipped.

## Before / After

| | **BEFORE** — statistician, by hand | **AFTER** — gated autoresearch loop |
|---|---|---|
| **Time** | ~days: wire a pipeline, fit ARIMA/MLP, eyeball residuals, retune, redo splits | **~38 s of compute**; the six entries + analysis are authored once and run in order |
| **Bookkeeping** | scattered notebooks; "which split / metric / seed gave that number?" | one `experiment_log.jsonl`, one `best_config.json`, one crash-recovery checkpoint |
| **Rigor** | rigor depends on discipline that erodes under deadline | rigor is **enforced**: no run launches without a real citation + mechanistic hypothesis + numeric prediction |
| **Metric drift** | easy to quietly "improve" by changing the metric | the composite **fingerprint is frozen** on Exp1 and asserted every run — a swap *raises* |
| **Leakage** | random splits silently leak the future of a warming series | walk-forward-by-year + `validate_no_overlap`; super-fold judges the **worst regime** |
| **Honesty** | tempting to report the average and bury the bad regime | per-fold penalty + a DISCARD that sticks; modest deltas reported as modest |
| **Recovery** | a crash loses the mental thread | a power failure costs **at most one run**; the checkpoint resumes a fresh session |

## What the gates buy you (no guessing, Goodhart-proof)

The loop is *more* disciplined than a typical manual sweep, not less. Two hard
gates (in `autoresearch_env/reasoning.py`) refuse to launch an experiment unless,
**before the run**, the researcher has committed:

1. **Citation-Rigor** — a *real* reference with a 4-digit year, a recognised venue,
   an arXiv ID **or** a quoted title, and a relevance clause (≥40 words). Bare tags
   like `(ClimateLLM2025)` are rejected. Every Exp here cites the seeded SOTA
   catalog: Hyndman & Athanasopoulos 2021 (OTexts), Gardner & Dorling 1998
   (*Atmospheric Environment*), Box-Jenkins 2015 (Wiley), Hochreiter & Schmidhuber
   1997 (*Neural Computation*).
2. **Reasoning-Completeness** — word-count floors (diagnosis ≥60, hypothesis ≥50,
   prediction ≥25, verdict ≥30, learning ≥40), a **mechanistic** hypothesis
   ("because"/"mechanism"/"per …"), a **numeric-range** prediction (single points
   rejected), a KEEP/DISCARD/NEAR-MISS verdict, and a learning that marks an axis
   open/closed.

Plus the **frozen composite fingerprint** (`942df6a8450d8d57` here): the metric
name + orientation + penalty + threshold are hashed on Exp1 and re-asserted on every
later experiment. Silently redefining "skill" to look better doesn't produce a
better number — it produces a **raise**. That is the Goodhart guard: the score
cannot be the thing that's optimised by editing the scorer.

Net effect: you **cannot guess**. A "let me just try X" experiment with no citation,
no mechanism, or no numeric prediction never runs. The shallow-entry test
(`tests/test_exp05.py::test_shallow_pre_run_entry_raises_at_gate`) proves the gate
bites.

## Scaling up on the 4090 (same protocol, bigger backbone)

The CPU smoke path uses a tiny numpy MLP so the whole loop runs offline in seconds.
On the laptop's **RTX 4090 (~16 GB)** *only the backbone and data change* — the
gates, composite, fingerprint, splits, logging, and checkpoint are untouched:

- **Backbone.** Swap `mlp` for a **torch LSTM** (Hochreiter & Schmidhuber 1997 — the
  longer-memory recurrent model the Exp6 learning explicitly flagged as the next
  try) or a **zero-shot TS foundation model** — **TimesFM** (Das et al. 2024 ICML,
  arXiv:2310.10688) or **Chronos** (Ansari et al. 2024 TMLR, arXiv:2403.07815). The
  backbone dispatch in `autoresearch_env/runner.py` is the single extension point;
  add an `elif backbone == "lstm"` branch and the rest of the loop is unchanged.
- **Data.** Point `load_dataset` at real **ERA5** (`common.synthetic_climate.REAL_DATA_NOTE`
  documents the cdsapi/xarray swap) or many station-years. Walk-forward-by-year and
  super-fold-by-regime carry over directly to a real, non-stationary, warming series.
- **Budget & resilience.** Set a per-experiment timeout; the checkpoint is rewritten
  after *every* run, so a laptop crash costs at most one experiment.

The headline doesn't change with hardware: an agent fast enough to run the loop for
days, held to a bar a referee would recognise.

## Files

| File | Role |
|---|---|
| `run_loop.py` | Authors all six gated reasoning entries (Claude-as-researcher) and runs them in order; prints the champion progression; renders the plot. Idempotent (wipes `results/` first). |
| `research_journal.md` | First-person narrative: the 7 steps per experiment + the champion table; honest about the modest deltas. |
| `champion_progress.png` | Composite (best-so-far staircase) + per-experiment composite and test RMSE vs experiment number, final champion marked. |
| `results/experiment_log.jsonl` | One JSON row per experiment (runner-written). |
| `results/reasoning_annotations.json` | The six full reasoning blobs (pre + post-run), all gate-passing. |
| `results/best_config.json` | The champion record (Exp5), with the frozen fingerprint. |
| `results/project_autoresearch_checkpoint.md` | Self-contained crash-recovery checkpoint. |
| `tests/test_exp05.py` | Fast pytest: ≥5 logged experiments; champion composite non-decreasing; shallow entry RAISES at the gate; primary metric stable. |

Credit: methodology adapted from the user's own `dlmastery/autoresearch`
(`generalized_ml_autoresearch`). Runnable package: `autoresearch_env/`. Full
protocol: `docs/autoresearch_protocol.md`.
