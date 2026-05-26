---
name: env-autoresearch-setup
description: >
  Interactive wizard that scaffolds an environmental-statistics AUTORESEARCH project
  (the env-stats adaptation of /ml-autoresearch-setup). Use when the user says
  "/env-autoresearch-setup", "set up autoresearch for my env-stats project",
  "bootstrap an environmental autoresearch loop", or wants to start an agent-driven
  research loop on climate / biodiversity / remote-sensing / hydrology / air-quality
  data. Walks through task type, dataset, metrics, split, and backbones, then writes
  a config, a seed reasoning entry, and a folder skeleton, and explains the 7-step loop.
version: 0.1.0
arguments:
  - name: project_root
    required: false
    description: Absolute path to the new project root (defaults to cwd/env-autoresearch-project).
triggers:
  - "/env-autoresearch-setup"
  - "set up autoresearch for env stats"
  - "bootstrap environmental autoresearch"
  - "start an env-stats research loop"
---

# env-autoresearch-setup — environmental-statistics autoresearch wizard

You are the setup-wizard skill. Walk the user through the steps below (use
`AskUserQuestion` for interactive choices, `Read`/`Write` for files). The output is
a runnable autoresearch project bootstrap: a `config.yaml`, a seed reasoning entry,
and a folder skeleton — plus a clear explanation of the 7-step loop the agent will
then run.

This is the env-stats adaptation of the source repo's `/ml-autoresearch-setup`
(`dlmastery/autoresearch → generalized_ml_autoresearch`). The methodology — Claude
Code as the researcher, hard gates, pluggable backbones/splits, a composite metric —
is preserved; only the task types, datasets, metrics, and splits are specialized for
environmental statistics. The reference implementation lives in this repo at
`autoresearch_env/`.

## Hard rules (carry over from the source methodology and CLAUDE.md)

- **No experiment runs before its reasoning entry exists.** The seed entry written in
  Step 7 is empty of results by design; Exp1's pre-run reasoning must be authored
  BEFORE launching (Diagnose → Cite → Hypothesize → Predict).
- **Citation gate.** Every backbone/recipe choice must cite a verifiable anchor (see
  the SOTA catalog in `references/env_sota_catalog.md`). No invented papers, dates, or
  version numbers — flag anything unconfirmed as `(verify)`.
- **Composite-metric fingerprint.** Record the composite formula at setup so a later
  mid-project rewrite (Goodhart risk) is detectable. Changing it later requires an
  explicit `RULE_CHANGE` note.
- **Leakage discipline is the default, not an option.** Temporal order and spatial
  grouping are honored by the chosen split (Step 4). When in doubt, the stricter split
  wins. Pair this skill with `env-stats-validate` before trusting any result.

## Step 1 — Task type

Ask which environmental-statistics task this is:

- `forecasting` — predict a future environmental series/field (temperature, streamflow,
  PM2.5). Order is meaningful; look-ahead is a leakage risk.
- `classification` — land cover, change/no-change, regime label, species presence.
- `evt` — extreme-value analysis: GEV/GPD fits, return levels, block maxima / POT,
  trend-in-extremes. (Rigor-heavy; metric is about tail fit + return-level CIs.)

Also record: is there a natural grouping variable that must NOT cross a split
(basin/gauge ID, grid cell, climate region, station)? If yes, a spatial-group split is
mandatory in Step 4. Save a one-line `domain_description` (e.g. "PM2.5 nowcasting at
OpenAQ stations").

## Step 2 — Dataset

Pick the data source. For each, point the user at the real-fetch path AND the offline
synthetic stand-in (so the loop runs with no API key / no network for prototyping):

| Choice | Real source (see `climate-data-fetch` skill) | Offline synthetic stand-in (this repo) |
|---|---|---|
| ERA5 | `cdsapi` + `xarray` (Copernicus CDS) | `common/synthetic_climate.py` (`daily_temperature`, `gridded_temperature_field`) |
| GBIF | `pygbif` | `common/synthetic_biodiversity.py` (`citizen_comments`) |
| Sentinel-2 | `sentinelhub` / STAC + `rasterio` | `common/synthetic_remote_sensing.py` (`multispectral_patches`, `change_pair`) |
| CAMELS | dataset download (basin attributes + forcings) | `common/synthetic_climate.py` (use streamflow-style series) |
| OpenAQ | OpenAQ API | `common/synthetic_climate.py` (`daily_precipitation` / station-style series) |
| synthetic | — | the matching `common/synthetic_*.py` generator |

Default the user to the synthetic generator first ("runs-anywhere"): deterministic
seeds, no keys, no GPU. Record target column/variable, feature columns, and the
group/time column.

## Step 3 — Primary + composite metric

Pre-populate the primary metric by task type, using the implementations already in
`common/metrics.py` (cite them by name so the agent reuses them, not re-derives them):

- `forecasting` → **latitude-weighted RMSE** (`latitude_weighted_rmse`, for gridded
  fields, cos(lat) weighting — the ERA5-benchmark convention) or plain **RMSE**
  (`rmse`) for point series; **ACC** (`anomaly_correlation`) when a skill/correlation
  framing is wanted (operational target ACC > 0.6).
- `classification` → **macro-F1 / accuracy** (`classification_report_simple`).
- `evt` → return-level error / tail goodness-of-fit; report return-level **confidence
  intervals** (this is a rigor requirement, not a nicety).

Always pair the primary metric with a **skill score** vs a baseline
(`skill_score(model, reference)`): 1 = perfect, 0 = no better than the reference,
<0 = worse. Note higher- vs lower-is-better.

**Composite metric** (preserved from the source loop, default):

```
composite = min(val_primary, test_primary) - penalty * n_below_threshold_folds
```

Offer alternatives (smoother `mean(...)`; overfit-penalizing
`test - penalty*max(0, test - val)`). Record `penalty_weight`, `below_threshold`, and
store a `composite_fingerprint` (a hash/string of the formula) so rewrites are caught.

## Step 4 — Split (leakage-safe by construction)

Pick the split to match the task and grouping. These are the env-stats specializations:

- `walk_forward` — forecasting on time series: fixed train/val/test windows that roll
  forward, with **purge + embargo** sized to the label horizon (never let the future
  leak). For point series, `common/synthetic_climate.time_split` gives the chronological
  no-shuffle baseline.
- `spatial_group` — spatial GroupKFold: whole basins / grid cells / stations are held
  out together so nearby (autocorrelated) points never straddle train and test.
- `super_fold` — union of multiple regime windows (e.g. by climate regime / ENSO phase
  / wet-vs-dry years); the env analogue of the source repo's super-fold.

Record n_splits, val_fraction, purge, embargo, and (for super-fold) the regime window
definitions.

## Step 5 — Backbones (from the env SOTA catalog)

Auto-suggest a tiered backbone list from `references/env_sota_catalog.md`, filtered by
task type. Every selected backbone must carry its verifiable anchor (citation gate):

- **Tier 1 — classical baselines:** climatology / persistence (forecasting), ARIMA/STL,
  Mann-Kendall + GEV/GPD (EVT), logistic / Random-Forest-on-indices (classification).
  These are the "BEFORE" and the skill-score reference.
- **Tier 2 — modern / foundation:** time-series foundation models for **zero-shot**
  forecasting — TimesFM, Chronos, MOMENT, Moirai (no per-series training); small
  CNN/LSTM/TCN on the 4090 for fields/imagery; EagleVision-style detection for EO.
- **Tier 3 — GBMs (keep as separate backbones):** xgboost, lightgbm, catboost — do not
  merge them into one.

The user can add/remove. Keep at least one Tier-1 baseline so the skill score is
meaningful.

## Step 6 — Hardware / budget

GPU VRAM (GB), CPU cores, cores reserved for the runner, time budget per experiment,
N experiments per backbone (default 50), and multi-seed policy (default: 3-seed median
before declaring a champion). Note: the runs-anywhere baseline must stay CPU-only and
keyless; GPU is for Tier-2/3 only.

## Step 7 — Write the scaffold

Create the folder skeleton and seed files under `<project_root>`:

```
<project_root>/
  config.yaml                      # all wizard answers (machine-readable)
  data/                            # cache for fetched/synthetic data
  reasoning/
    reasoning_entries.jsonl        # seed entry only (NO results yet)
  results/                         # committed metrics, plots, tables
  winners/                         # champion archive folders
  memory/
    checkpoint.md                  # crash-recovery checkpoint + 7-step reminder
  research_journal.md              # "## Project initialized <ISO date>"
```

Write `config.yaml` from `references/config_template.yaml`, resolving every value from
the wizard answers (task_type, dataset, primary_metric, composite formula + fingerprint,
split + params, backbones with citations, hardware/budget).

Write the **seed reasoning entry** to `reasoning/reasoning_entries.jsonl` using
`references/seed_reasoning_entry.json` as the shape — one entry, `phase: "pre-run"`,
empty `results`, `experiment_id: "exp1"`, capturing the Diagnose/Cite/Hypothesize/
Predict fields the agent must fill before launching.

Show a diff/preview before writing. After writing, tell the user the exact first action:
**author Exp1's pre-run reasoning (Diagnose → Cite → Hypothesize → Predict) in
`reasoning/reasoning_entries.jsonl` BEFORE running anything.**

## The 7-step research loop (explain this to the user)

The agent (Claude Code as the researcher) then runs this loop, one experiment per cycle:

1. **Diagnose** — read the current best result + diagnostics; identify the single
   biggest weakness (bias, miscalibration, poor tail fit, leakage suspicion).
2. **Cite** — find a verifiable method/anchor that addresses it (citation gate; use the
   SOTA catalog). No anchor → no experiment.
3. **Hypothesize** — state the change and why it should help, in one sentence.
4. **Predict** — write the expected metric movement BEFORE running (this is the
   reasoning entry; the runner refuses to launch without it).
5. **Execute** — run exactly ONE experiment (one backbone/config change).
6. **Analyze** — compute primary + composite + skill score on the leakage-safe split;
   compare prediction vs outcome; run the `env-stats-validate` checklist.
7. **Checkpoint** — update `memory/checkpoint.md`, archive a winner if it is a new
   global best, and commit (power-failure recovery). Then loop.

## Non-negotiable gates (verify before finishing)

- [ ] `config.yaml` has no unresolved placeholders.
- [ ] Every selected backbone cites an anchor from the SOTA catalog (or is marked `(verify)`).
- [ ] The composite formula and its fingerprint are recorded.
- [ ] The split honors temporal order / spatial grouping (no obvious leakage path).
- [ ] `reasoning_entries.jsonl` has exactly the seed entry with NO results.
- [ ] `memory/checkpoint.md` states the full 7-step loop and "author Exp1 reasoning before launching".

## Failure modes to warn about

- "I have no dataset yet" → start on the matching `common/synthetic_*.py` generator;
  it is a deliberate, documented stand-in, not a crutch (swap-in notes are in each file).
- "Can I change the composite metric later?" → it is a Goodhart risk; require a
  `RULE_CHANGE` note in the checkpoint and re-fingerprint.
- "Can I use a random/shuffled split for a time series or spatial data?" → No. That
  leaks the future / leaks across autocorrelated neighbors. Use `walk_forward` /
  `spatial_group` / `super_fold`.
- "Can I skip the pre-run reasoning?" → No. It is the prediction-before-result gate.
