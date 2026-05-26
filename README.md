# environment_stats_talk

### From Data to Discovery — transforming Environmental Statistics with Claude Code & AI-for-Science
**Companion repository for the talk at the International Environment Statistics Institute conference, Mexico City, 7–11 December 2026 — session _"Modern Approaches to Environmental Statistics."_**

---

## The thesis (one sentence)

> Environmental statistics is being transformed by AI coding agents and AI-for-science models — and the gap is best shown as **BEFORE** (the traditional workflow: a person, specialist tools, days–weeks) vs **AFTER** (Claude Code + an AI-for-science model: the same result in minutes, with richer artifacts and preserved statistical rigor).

This repo is the **living proof behind the slides.** Every claim on a slide points to a folder here with runnable code and **real, committed results**.

## What makes this credible (and different)

- **Before/After, with receipts.** Each topic ships a `before/` script (traditional method), an `after/` script (agentic / AI-for-science), and a `results/` folder with the actual metrics, plots, and tables produced on this machine.
- **Runs anywhere, immediately.** The headline results are produced from **synthetic + small public data with no API keys and no GPU required** (deterministic seeds → reproducible). Every experiment also documents a **real-data variant** and, where relevant, a **4090 GPU variant**.
- **Real hardware.** Heavy variants were exercised on an **NVIDIA RTX 4090 (Laptop, ~16 GB)** — the same class of hardware an individual scientist can own.
- **Zero-hallucination discipline.** This is for *scientists*. We build only on **verifiable** papers/tools/datasets; every speculative or future-dated claim from the source material is quarantined in [`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md) and never presented as fact.
- **Autoresearch.** The flagship demo adapts a Karpathy-style **autonomous research loop where Claude Code is the researcher** (from `dlmastery/autoresearch`'s `generalized_ml_autoresearch`) to environmental statistics, running on the 4090.

## Repository map

| Path | What's inside |
|------|---------------|
| [`CLAUDE.md`](CLAUDE.md) | The full project brief & operating rules (read first if you're an agent). |
| [`ledgers/`](ledgers/) | `INSIGHTS.md`, `TODO.md` (master task list), `ITEMS.md` (papers/tools/data/benchmarks), `CITATIONS-TO-VERIFY.md`. |
| [`docs/`](docs/) | `taxonomy.md`, `before_after.md`, `zero_to_hero.md`, `autoresearch_protocol.md`. |
| [`docs/talk/`](docs/talk/) | Slide outline + speaker notes, one-page handout, live-demo script. |
| [`common/`](common/) | Shared synthetic-data generators + plotting helpers (no network). |
| [`experiments/`](experiments/) | The before/after experiments, each self-contained with committed results. |
| [`autoresearch_env/`](autoresearch_env/) | Environmental-statistics adaptation of the autoresearch loop. |
| [`skills/`](skills/) | Claude Code skills (autoresearch setup, data fetch, statistical-rigor validation). |

## Experiments (each is a BEFORE → AFTER pair)

| # | Topic | Before | After | Headline artifact |
|---|-------|--------|-------|-------------------|
| 01 | Climate time-series forecasting | persistence / ARIMA | LSTM/TCN on 4090 (+ foundation-model zero-shot) | RMSE/ACC table + forecast plots |
| 02 | Climate extremes & trends | hand-coded Mann-Kendall + GEV | Claude-Code pipeline + rigor checklist | return-level plot + trend CIs |
| 03 | Biodiversity from text | regex/keyword extraction | structured-LLM extraction → interaction network | precision/recall + network graph |
| 04 | Remote-sensing land cover | Random Forest on spectral indices | CNN on 4090 (+ embeddings) | accuracy/F1 + change map |
| 05 | **Autoresearch loop (flagship)** | one model, by hand | Claude Code runs N experiments autonomously | champion archive + research journal |

See [`ledgers/TODO.md`](ledgers/TODO.md) for the full, expanding task list.

## Quick start

```bash
pip install -r requirements.txt
# Run any experiment's committed-results path (no keys/GPU needed):
python experiments/01_climate_timeseries_forecast/run_before_after.py
# Open the aggregated scoreboard when built:
#   RESULTS.md  (top-level before/after "100x" table)
```

## Status

🚧 Under active construction by Claude Code (multi-agent build). Documentation and code are written and unit-tested **before** experiments are run. Progress is checkpointed to this public repo periodically (power-failure recovery). See [`ledgers/TODO.md`](ledgers/TODO.md).

## Credits & licensing

Autoresearch methodology adapted from `dlmastery/autoresearch` (`generalized_ml_autoresearch`). Source-material talk content originates from a research-planning transcript; all factual claims here are independently filtered for verifiability. License: MIT (see `LICENSE` once added).
