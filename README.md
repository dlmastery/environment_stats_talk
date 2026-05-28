# environment_stats_talk

### From Data to Discovery — transforming Environmental Statistics with Claude Code & AI-for-Science
**Companion repository for the talk at the International Environment Statistics Institute conference, Mexico City, 7–11 December 2026 — session _"Modern Approaches to Environmental Statistics."_**

[![tests](https://img.shields.io/badge/tests-13%20components%20green-brightgreen)](run_all_tests.py) [![experiments](https://img.shields.io/badge/before%2Fafter%20experiments-11-blue)](RESULTS.md) [![hardware](https://img.shields.io/badge/runs%20on-CPU%20or%20RTX%204090-orange)](#hardware--reproducibility) [![rigor](https://img.shields.io/badge/claims-web--verified-success)](ledgers/CITATIONS-TO-VERIFY.md)

---

## The thesis (one sentence)

> Environmental statistics is being transformed by AI coding agents and AI-for-science models — and the gap is best shown as **BEFORE** (the traditional workflow: a person, specialist tools, days–weeks) vs **AFTER** (Claude Code + an AI-for-science model: the same result in minutes, with richer artifacts and preserved statistical rigor).

This repo is the **living proof behind the slides.** Every claim on a slide points to a folder here with runnable code and **real, committed results**.

## Table of contents

- [Why this is credible (and different)](#why-this-is-credible-and-different)
- [Quickstart](#quickstart)
- [The 11 before/after experiments](#the-11-beforeafter-experiments)
- [Repository map](#repository-map)
- [Documentation index](#documentation-index)
- [The autoresearch loop (flagship)](#the-autoresearch-loop-flagship)
- [The talk kit](#the-talk-kit)
- [Hardware & reproducibility](#hardware--reproducibility)
- [The honest-results philosophy](#the-honest-results-philosophy)
- [Status](#status)
- [Credits & license](#credits--license)

## Why this is credible (and different)

- **Before/After, with receipts.** Each topic ships a `before/` script (traditional method), an `after/` script (agentic / AI-for-science), and a `results/` folder with the actual metrics, plots, and tables produced on this machine. The scoreboard is [`RESULTS.md`](RESULTS.md).
- **Runs anywhere, immediately.** Headline results come from **synthetic + small data with no API keys and no GPU** (deterministic seeds → reproducible). Every experiment also documents a **real-data** variant and, where relevant, a **4090 GPU** variant.
- **Honest by design.** Across 9 experiments the verdicts are mixed *on purpose* — clean AI wins where structure rewards it, ties/classical-wins where deserved (e.g. **SARIMA beats the LSTM at 1-step**). That mix is the point. See [the philosophy](#the-honest-results-philosophy).
- **Zero-hallucination discipline.** Built only on **verifiable** papers/tools/datasets; every claim was **web-verified** ([`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md)) and unverified performance stats stay flagged.
- **Autoresearch.** The flagship adapts a Karpathy-style **autonomous research loop where Claude Code is the researcher** (from `dlmastery/autoresearch`) to environmental statistics, with hard citation/reasoning gates.

## Quickstart

```bash
pip install -r requirements.txt          # numpy/pandas/sklearn/scipy/statsmodels/matplotlib/torch
python experiments/01_climate_timeseries_forecast/run_before_after.py --quick   # first before/after in minutes
python run_all_tests.py                   # all 13 test components (CPU)
```

New here? Start with **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** → then **[docs/EXPERIMENTS_INDEX.md](docs/EXPERIMENTS_INDEX.md)**.

## The 11 before/after experiments

Full numbers and honest verdicts in **[RESULTS.md](RESULTS.md)**. Each row is self-contained under `experiments/`.

| # | Topic | Before → After | Headline (real, on this machine) | GPU |
|---|-------|----------------|----------------------------------|-----|
| [01](experiments/01_climate_timeseries_forecast) | Climate forecasting | persistence/SARIMA → LSTM | LSTM skill +0.32 @14-day; **SARIMA wins @≥7-day** (honest) | 4090 |
| [02](experiments/02_extreme_value_trends) | Climate extremes & trends | empirical → GEV + bootstrap CIs | 100-yr level **55.8 mm [46.7–80.2]**; flags its own stationarity violation | — |
| [03](experiments/03_biodiversity_text_extraction) | Biodiversity from text | regex → structured extraction | recall **10% → 100%** (synthetic ceiling, caveated) | — |
| [04](experiments/04_remote_sensing_landcover) | Land cover + change | RF on indices → CNN on bands | easy: tie; **hard-mode: RF 0.64 → CNN 0.998 (+0.36)** | 4090 |
| [05](experiments/05_autoresearch_climate) | **Autoresearch loop (flagship)** | by hand → gated 6-experiment loop | champion **+8.0% skill** in ~38 s; gates block guesses | — |
| [06](experiments/06_spatial_interpolation) | Spatial interpolation | kriging → RF + covariate | **2.91 → 2.20 RMSE** (kriging keeps a variance surface) | — |
| [07](experiments/07_air_quality_nowcast) | Air-quality PM2.5 | persistence/ARIMA → GBM + weather | **17.4 → 12.1 RMSE**, spike-F1 0.74 → 0.85 | — |
| [08](experiments/08_hydrology_streamflow) | Hydrology streamflow | linear/bucket → LSTM | **NSE 0.14 → 0.70** (+0.56) | 4090 |
| [09](experiments/09_bayesian_vs_amortized) | Bayesian: hierarchical inference | MCMC (from-scratch MH) → amortized MLP | **~1451× faster at scoring**, coverage parity, wider intervals (honest tradeoff) | — |
| [10](experiments/10_species_distribution) | Species distribution modeling | logistic GLM → calibrated GBM | AUC 0.71 → 0.73; **suit-corr 0.70 → 0.88** (+0.19) | — |
| [12](experiments/12_conformal_uncertainty) | Uncertainty quantification | normal-theory PIs → conformal | calibration gap **0.033 → 0.004 (~7×)**, narrower bands | — |
| [00](experiments/00_foundation_models_climate) | **Domain-specific weather foundations** | the talk's anchors, actually run | GraphCast (real ERA5 6-hr) · NeuralGCM (real ARCO-ERA5 4-day) · Pangu · FourCastNet · ClimaX — see [`docs/FOUNDATION_MODELS.md`](docs/FOUNDATION_MODELS.md) | 4090 / CPU |

## Repository map

```
environment_stats_talk/
├── README.md               ← you are here
├── CLAUDE.md               full project brief & operating rules (read first if you're an agent)
├── RESULTS.md              the before/after scoreboard (real numbers)
├── CONTRIBUTING.md         the before/after experiment contract (how to add one)
├── requirements.txt        deps (+ numpy-2 ABI pins)
├── run_all_tests.py        runs all 13 test components (one subprocess each)
├── common/                 shared synthetic-data generators + env metrics + headless plotting
│   ├── synthetic_climate.py  synthetic_biodiversity.py  synthetic_remote_sensing.py
│   ├── synthetic_hydrology.py  synthetic_spatial.py  synthetic_airquality.py
│   ├── metrics.py  plotting.py  tests/
├── experiments/            the 11 before/after experiments (each: before/ after/ run_before_after.py results/ tests/ README.md)
├── autoresearch_env/       env-stats adaptation of the Karpathy-style loop (gates, composite, splits, runner)
├── skills/                 Claude Code skills: env-autoresearch-setup, climate-data-fetch, env-stats-validate
├── docs/                   taxonomy, before/after, zero-to-hero, tutorials, guides, glossary, FAQ, talk kit
└── ledgers/                INSIGHTS, ITEMS, CITATIONS-TO-VERIFY, TODO, CHECKPOINT
```

## Documentation index

**Start / learn**
- [GETTING_STARTED.md](docs/GETTING_STARTED.md) — install, first run, no-GPU path, troubleshooting
- [EXPERIMENTS_INDEX.md](docs/EXPERIMENTS_INDEX.md) — all 9 experiments, run commands, verdicts
- [TUTORIAL_run_an_experiment.md](docs/TUTORIAL_run_an_experiment.md) — deep Exp01 walkthrough
- [TUTORIAL_autoresearch.md](docs/TUTORIAL_autoresearch.md) — drive the gated loop yourself
- [GLOSSARY.md](docs/GLOSSARY.md) — stats ↔ ML bridge terms

**Concepts / framing**
- [FOUNDATION_MODELS.md](docs/FOUNDATION_MODELS.md) — runnability matrix for the SOTA foundation models the transcript anchors on (TimesFM, Chronos, MOMENT, Moirai, GraphCast, NeuralGCM, Pangu, FourCastNet, ClimaX, CLLMate, WeatherNext, ClimateLLM, Prithvi, EagleVision) — what we actually ran, what we couldn't, why
- [taxonomy.md](docs/taxonomy.md) — environmetrics taxonomy + where AI fits (the talk's anchor map)
- [before_after.md](docs/before_after.md) — the BEFORE vs AFTER framework
- [zero_to_hero.md](docs/zero_to_hero.md) — the L1→L4 Claude-Code ladder
- [autoresearch_protocol.md](docs/autoresearch_protocol.md) — the env-stats autoresearch protocol

**Guides / practice**
- [GUIDE_using_claude_code_for_envstats.md](docs/GUIDE_using_claude_code_for_envstats.md) — the art of telling the agent what to do
- [GUIDE_adapt_to_real_data.md](docs/GUIDE_adapt_to_real_data.md) — swap synthetic → ERA5/GBIF/Sentinel-2/CAMELS/OpenAQ/SRTM
- [CONTRIBUTING.md](CONTRIBUTING.md) — add a new before/after experiment
- [FAQ.md](docs/FAQ.md) · [ADDENDUM_methodology.md](docs/ADDENDUM_methodology.md) — honest Q&A + methodology/threats-to-validity

**Talk kit** → [docs/talk/](docs/talk/): `outline.md` (16-slide notes), `deck.pptx` (20 slides w/ real figures), `handout.md`, `demo_script.md`.

## The autoresearch loop (flagship)

`autoresearch_env/` adapts the user's own `dlmastery/autoresearch` (`generalized_ml_autoresearch`) to environmental statistics. **Claude Code is the outer-loop researcher**, running the 7-step loop per experiment — **Diagnose → Cite → Hypothesize → Predict → Execute → Analyze → Checkpoint** — behind two hard gates:

- **Citation Rigor** (a real author/year/venue + arXiv/title + relevance note), and
- **Reasoning Completeness** (a *mechanistic* hypothesis + a *numeric* prediction range).

A frozen composite-metric fingerprint guards against Goodhart-style mid-project rewrites; a champion is archived only when the composite improves; a crash-recovery checkpoint is rewritten after every experiment. [Exp05](experiments/05_autoresearch_climate) ran it for real: a gated 6-experiment loop in ~38 s, champion **+8.0 % skill**, with the gates provably blocking a shallow "just try it" entry (proven by a test) and the agent even *rejecting* an obvious-but-wrong idea. See [docs/TUTORIAL_autoresearch.md](docs/TUTORIAL_autoresearch.md).

## Hardware & reproducibility

- Built and run on **Windows 11, Python 3.12, NVIDIA RTX 4090 Laptop GPU (~16 GB), CUDA 12.4, torch 2.6** — the class of hardware an individual scientist can own.
- **No GPU or API keys are required** for the committed results: everything runs on synthetic + small data with deterministic seeds. The 4090 only speeds up Exp01/04/08.
- `pip install -r requirements.txt` (note the **matplotlib ≥ 3.9** and **statsmodels ≥ 0.14.4** pins — earlier versions crash under numpy 2.x).
- `python run_all_tests.py` runs all 11 components (each in its own process to keep the per-experiment `before`/`after` packages from colliding).

## The honest-results philosophy

"AFTER" means *Claude Code + an AI-for-science model wrote, ran, and benchmarked the workflow.* The headline is the collapse in **human effort** (days of bespoke coding → one command) plus the **rigor gates** a rushed human often skips — **not** "a neural net always wins." So the repo keeps the losses visible: SARIMA beats the LSTM at 1-step (Exp01); RF ties the CNN on easy land cover (Exp04); kriging keeps an uncertainty surface RF lacks (Exp06). The value is that the agent lets you **find out fast, fairly, and reproducibly** — with the human owning the question, assumptions, multiple-testing, uncertainty, and sign-off ([`skills/env-stats-validate`](skills/env-stats-validate)). The "100× faster" claim is defined precisely in [docs/FAQ.md](docs/FAQ.md) (person-time + rigor, not a wall-clock multiplier).

## Status

✅ **v1++ complete.** Documentation + code + unit tests were written and verified **first** (by 5 waves of parallel SME agents), then the experiments were run for real. **11 before/after experiments, 13 green test components, a 20-slide deck (real charts embedded), a presenter demo-walkthrough, 14+ docs + tutorials/guides, 3 skills, the autoresearch loop, and web-verified citations.** Progress is checkpointed to this public repo after every unit (power-failure recovery): see [ledgers/CHECKPOINT.md](ledgers/CHECKPOINT.md) and [ledgers/TODO.md](ledgers/TODO.md).

## Credits & license

Autoresearch methodology adapted from **`dlmastery/autoresearch`** (`generalized_ml_autoresearch`). Source-material talk content originates from a research-planning transcript; all factual claims here are independently filtered for verifiability ([`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md)). License: MIT.
