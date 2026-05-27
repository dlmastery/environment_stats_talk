# Contributing — the before/after experiment contract

This repo is a companion to a conference talk on *Modern Approaches to Environmental
Statistics*. Its organising principle is **before/after**: every topic is a BEFORE
(traditional method) versus AFTER (agentic / AI-for-science method) pair, each with real,
committed artifacts and an honest verdict. This document is the **contract** every
experiment must satisfy so the repo stays reproducible, rigorous, and credible to a
skeptical audience.

Read [`CLAUDE.md`](CLAUDE.md) (the project brief and operating rules) and
[`RESULTS.md`](RESULTS.md) (the scoreboard) first. The methodology rationale is in
[`docs/ADDENDUM_methodology.md`](docs/ADDENDUM_methodology.md).

---

## The non-negotiable rules

1. **Runs-anywhere first.** Headline results must be reproducible from **synthetic +
   small public data, with no API keys, no network, and no GPU**, using deterministic
   seeds. Real-data and GPU variants are documented *separately* (see
   [`docs/GUIDE_adapt_to_real_data.md`](docs/GUIDE_adapt_to_real_data.md)), never required
   to reproduce the committed numbers. Optional real-data dependencies stay **commented
   out** in [`requirements.txt`](requirements.txt).

2. **Verifiable only — zero hallucination.** Build only on real, locatable papers/tools/
   datasets. Never invent a statistic, date, paper, or model version. Any speculative or
   future-dated claim goes to [`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md)
   and is **never** stated as fact in a README, slide, or code comment. Mark anything you
   are unsure of with **(verify)**. Quote only the repo's own measured numbers, not
   transcript-style multipliers.

3. **Honest verdicts.** Report ties and classical-wins as plainly as AFTER-wins. Do not
   manufacture a synthetic accuracy win the data does not support; if the cheap baseline is
   already excellent, say the AFTER value is *effort/capability*, not accuracy.

4. **Statistical rigor preserved.** The AFTER side accelerates; it does not cut corners.
   Honour leakage discipline (temporal AND spatial), report an honest baseline + skill
   score, quantify uncertainty, and run the
   [`env-stats-validate`](skills/env-stats-validate/SKILL.md) checklist before reporting.

5. **Use `common/`, don't reinvent.** Synthetic data comes from `common/synthetic_*.py`;
   metrics from `common/metrics.py`; plots from `common/plotting.py`. Do not modify
   `common/` to make an experiment look better — its tests are part of the contract.

6. **Fast, CPU-only unit tests.** Every experiment ships tests that run on CPU in well
   under ~30 s of *compute* (cold-start library import may dominate wall time). Tests must
   be deterministic and need no key/network.

---

## Folder layout (the standard experiment skeleton)

Every experiment lives under `experiments/NN_short_name/` and follows this layout:

```
experiments/NN_short_name/
├── before/                  # traditional method(s) — pure, transparent, usually CPU
│   ├── __init__.py
│   └── <method>.py          # exposes run_before(...) (or the method's functions)
├── after/                   # agentic / AI-for-science method(s)
│   ├── __init__.py
│   └── <method>.py          # exposes run_after(...); auto-CUDA, CPU fallback
├── run_before_after.py      # orchestrator: runs both, writes results/, builds plots
├── results/                 # COMMITTED artifacts: metrics.json, *.png, summary.md
│   ├── metrics.json         # all numbers + config/provenance (machine-readable)
│   ├── <plots>.png          # the before/after figures
│   └── summary.md           # human-readable scorecard
├── tests/
│   ├── __init__.py
│   └── test_expNN.py        # fast, CPU, deterministic, no key/network
└── README.md                # the story + comparison table + how-to-run + real-data swap
```

Notes:
- `before/` and `after/` are **local packages** whose names intentionally repeat across
  experiments — that is why tests run **per-experiment in separate subprocesses** (see
  below). Always run scripts **from the repo root** so `import common` resolves.
- The flagship autoresearch experiment (Exp05) uses `run_loop.py` instead of
  `run_before_after.py` and adds `research_journal.md` + a results checkpoint; it reuses the
  `autoresearch_env/` package. Follow its README as the template for loop-style experiments.
- Some experiments add a small local helper module (e.g. `aq_metrics.py`, `hydro_metrics.py`,
  `forecaster.py`) when a metric/forecaster is shared between `before/` and `after/`.

## The README contract

Each experiment README must contain, in this spirit:

- **Task** — one paragraph: what is predicted/estimated and why it matters.
- **The before/after story** — what BEFORE codes by hand vs what AFTER adds, and *why* the
  data has the structure that makes the contrast real (not rigged by noise).
- **A comparison / results table** — the committed metrics (or explicit `_placeholder_`
  cells until a full run exists) plus illustrative person-time vs Claude-Code effort,
  labelled as *illustrative estimates of build effort, not measured benchmarks*.
- **An honest note** — where the baseline is already enough, the uncertainty tradeoff, or
  the synthetic ceiling, as applicable.
- **How to run** — `--quick` smoke run, default run, GPU/headline run, and the test command.
- **Swapping in real data** — the data contract to preserve and the real source (cross-link
  [`docs/GUIDE_adapt_to_real_data.md`](docs/GUIDE_adapt_to_real_data.md) and the
  [`climate-data-fetch`](skills/climate-data-fetch/SKILL.md) skill).
- **References** — described generically, with **(verify)** on anything not nailed down.

## The results contract

`results/metrics.json` is machine-readable and self-describing: it carries every reported
number, the AFTER−BEFORE deltas, and the **config/provenance** (seed, sizes, device,
flags) so a number can be traced to the run that produced it. `summary.md` is the
human-readable version of the same. Plots are committed PNGs. Commit the artifacts from a
deterministic run; if a metric awaits a full GPU pass, use an explicit `_placeholder_`
cell in the README rather than a guessed number.

## Tests & `run_all_tests.py`

- Add your experiment's test directory to the `COMPONENTS` list in
  [`run_all_tests.py`](run_all_tests.py), keeping the order **foundations → autoresearch →
  experiments** (lowest experiment number next). `run_all_tests.py` runs each component in
  its **own subprocess** to avoid `sys.modules` collisions between the repeated
  `before/`/`after/` package names.
- Run the whole suite: `python run_all_tests.py` (pass-through args work, e.g.
  `python run_all_tests.py -k expNN`).
- Run one experiment's tests directly (from the repo root):
  `python -m pytest experiments/NN_short_name/tests -q`.
- A headline test should assert the experiment's *claim* (e.g. "covariate model beats
  persistence on a held-out tail", "LSTM NSE > linear NSE", "conformal gap < normal-theory
  gap"), not just that code runs.

---

## How to add a new experiment (step by step)

1. **Pick the slot.** Choose the next `NN` and a short name; create
   `experiments/NN_short_name/`.

2. **Find or add a synthetic generator in `common/`.** Reuse an existing
   `common/synthetic_*.py` if one fits; otherwise add a new generator that (a) returns a
   clean, documented **data contract** matching the real source, (b) injects the *signal*
   the experiment must recover (so the before/after contrast is real), (c) is deterministic
   given a seed, and (d) documents the real-data swap (a `REAL_DATA_NOTE` / docstring note).
   Add generator tests under `common/tests/`.

3. **Write `before/`** — the honest traditional baseline(s). Pure, transparent, usually
   CPU-only. Expose a `run_before(...)` (or the method functions the orchestrator calls).

4. **Write `after/`** — the agentic / AI-for-science method. Expose `run_after(...)`;
   auto-select the device (CUDA when present, CPU fallback) so it is runs-anywhere. Keep any
   real-LLM / foundation-model / real-data path **optional and gated** (skips cleanly,
   never fabricates a number, when the key/package/encoder is absent).

5. **Write `run_before_after.py`** — the orchestrator: run both sides on a leakage-safe
   split, compute primary + skill metrics with `common/metrics.py`, write
   `results/metrics.json` (+ config/provenance) and `summary.md`, render plots with
   `common/plotting.py`. Support `--quick`, `--seed`, and the relevant size/epoch flags.

6. **Write `tests/test_expNN.py`** — fast, CPU, deterministic, no key/network; assert the
   data has the intended signal, both sides run leak-free, metrics are finite, and the
   headline claim holds on a held-out split.

7. **Write `README.md`** — to the README contract above (story, table, honest note, how to
   run, real-data swap, references with **(verify)**).

8. **Wire it up.** Add the test dir to `run_all_tests.py`; run
   `python -m pytest experiments/NN_short_name/tests -q` and `python run_all_tests.py`
   green; produce the committed `results/` from a deterministic run.

9. **Add it to the scoreboard.** Add a row to [`RESULTS.md`](RESULTS.md) with the honest
   verdict, and cross-link the real-data swap in
   [`docs/GUIDE_adapt_to_real_data.md`](docs/GUIDE_adapt_to_real_data.md).

10. **Validate before you call it done.** Run the
    [`env-stats-validate`](skills/env-stats-validate/SKILL.md) checklist; route any unsure
    citation to [`ledgers/CITATIONS-TO-VERIFY.md`](ledgers/CITATIONS-TO-VERIFY.md).

---

## Git / checkpointing

Commit + push after each meaningful unit (power-failure recovery). Keep commits scoped to
one experiment/doc where possible. Do not commit large fetched datasets — the runs-anywhere
path needs none, and real data is reproduced via the fetch snippets + provenance sidecars.
Credit the autoresearch methodology to the user's own `dlmastery/autoresearch`
(`generalized_ml_autoresearch`).
