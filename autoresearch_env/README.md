# autoresearch_env — env-stats AutoResearch loop (lightweight, self-contained)

A minimal, runs-anywhere adaptation of the user's own
[`dlmastery/autoresearch`](https://github.com/dlmastery/autoresearch)
(`generalized_ml_autoresearch`) loop, specialised for **environmental statistics**.

It keeps the parts that matter for the talk and drops everything heavyweight:

- the **7-step research loop** (Diagnose → Cite → Hypothesize → Predict → Execute →
  Analyze → Checkpoint), with **Claude Code as the outer loop**;
- the **two hard gates** — Citation Rigor + Reasoning-Blob Completeness — that refuse
  to run an experiment without a real citation, a *mechanistic* hypothesis, and a
  *numeric* prediction;
- the **composite metric** `min(val, test) − penalty · n_below_threshold` with a
  **frozen fingerprint** that detects a mid-project metric rewrite (Goodhart guard);
- env-flavored **splits**: walk-forward by year and super-fold by climate regime.

It **does not import** the source repo. It reuses *this* repo's `common` package for
synthetic data + metrics, so it runs on CPU with no API keys.

Credit: methodology adapted from `dlmastery/autoresearch` (`generalized_ml_autoresearch`).

## Files

| File | Role |
|---|---|
| `reasoning.py` | Citation-Rigor + Reasoning-Completeness gates; `ReasoningEntry`, `ReasoningAnnotationsFile`. |
| `composite.py` | `CompositeCalculator` — env composite metric + frozen-fingerprint check. |
| `splits.py` | `WalkForwardSplit` (by year) + `SuperFoldSplit` (by climate regime). |
| `runner.py` | One-experiment runner: gate → fingerprint → train backbone → log JSONL + champion + checkpoint. |
| `sota_catalog_env.yaml` | Vetted env-stats backbones/baselines with REAL citations (seeds the gate). |
| `tests/test_autoresearch_env.py` | Fast pytest covering the gate, the fingerprint, and the mini-loop. |

## How Claude Code drives the loop

There is no separate Python agent. **Claude Code is the researcher.** Each turn:

1. **Diagnose** — read `experiment_log.jsonl` + the checkpoint; find the champion's
   weakest fold/regime.
2. **Cite** — pick a real reference from `sota_catalog_env.yaml` (or the verified
   anchors in the repo's `CLAUDE.md` §6) that addresses the weakness.
3. **Hypothesize** — write a *mechanistic* hypothesis ("…because the mechanism is…").
4. **Predict** — write a *numeric range* for the expected composite/skill change.
5. Author the pre-run entry with `ReasoningAnnotationsFile.commit_pre_run(...)`.
   It **raises** if the gate fails — Claude cannot run a guess.
6. **Execute** ONE experiment: `run_experiment(config)` (or the CLI below).
7. **Analyze** vs the prediction; write `verdict` (KEEP/DISCARD/NEAR-MISS) +
   `learning` (axis open/closed, next try) via `commit_post_run(...)`.
8. **Checkpoint** — the runner already rewrote
   `project_autoresearch_checkpoint.md`; a fresh session can resume from it alone.

### Run one experiment

```bash
# from the repo root
python -m autoresearch_env.runner --config path/to/exp.yaml --description "MLP, 7 lags"
```

Programmatically:

```python
from autoresearch_env import run_experiment, ReasoningAnnotationsFile, ReasoningEntry

ann = ReasoningAnnotationsFile("results/reasoning_annotations.json")
ann.commit_pre_run(ReasoningEntry(experiment_num=1, diagnosis=..., citations=...,
                                  hypothesis=..., prediction=...))   # gate enforced here
record = run_experiment(config_dict)   # refuses to run if the gate is unmet
```

### Config shape

```yaml
paths: { results_dir: "experiments/01_climate_timeseries_forecast/results/autoresearch" }
primary_metric: lat_weighted_rmse        # or acc / skill_vs_persistence / rmse / mae
backbone: mlp                            # or persistence
backbone_config: { hidden: 16, epochs: 300, lr: 0.05, weight_decay: 1.0e-4 }
data: { n_years: 12, seed: 0, n_lags: 7, n_regimes: 3 }
split: { name: walk_forward, n_blocks: 5, n_initial: 2, gap: 1 }   # or super_fold
composite: { higher_is_better: false, penalty_weight: 0.5, below_threshold: -12.0 }
seed: 0
description: "baseline MLP, 7 lags"
```

## How Exp05 will use it on the 4090

Exp05 is the flagship: the loop running **autonomously** to improve the Exp01 climate
forecaster. On the laptop's RTX 4090 the only changes from the CPU smoke path are:

- **Backbones scale up.** Swap the numpy MLP for a torch LSTM / TimesFM / Chronos
  zero-shot backbone (catalog tiers 2–3). The runner's backbone dispatch is the one
  spot to extend; the gate, composite, splits, logging, and checkpoint are unchanged.
- **Data scales up.** Point `load_dataset` at real ERA5 (`common.synthetic_climate`'s
  `REAL_DATA_NOTE` documents the swap) or many years/stations; walk-forward-by-year
  and super-fold-by-regime carry over directly.
- **Budget.** Set a per-experiment timeout; checkpoint after *every* experiment so a
  power failure on the laptop loses at most one run (the repo's resilience rule).

The protocol — gate every experiment, freeze the composite, judge on the worst regime
and on future years — is identical at 4090 scale. See `docs/autoresearch_protocol.md`
for the full worked example and the env composite/split definitions.

## Tests

```bash
python -m pytest autoresearch_env/tests -q
```

Covers: the gate **rejects** a missing/shallow entry and **accepts** a complete one;
the composite **fingerprint detects** a changed metric; and a 2–3 experiment mini-loop
runs on synthetic data with the champion updating **only** when the composite improves.
