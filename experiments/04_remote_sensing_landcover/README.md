# Experiment 04 — Remote-sensing land cover & change detection (BEFORE vs AFTER)

**Task.** Classify 5-band multispectral image patches into five land-cover
classes — `water, forest, cropland, urban, bare` — and run a small
change-detection mini-task (deforestation: forest → bare/cropland) on a
before/after image pair.

This is the canonical environmental-remote-sensing workflow, set up as a
**before/after** pair so a skeptical audience can see exactly what changes when an
agentic coder (Claude Code) and modern DL replace the hand-rolled GIS pipeline —
and, just as importantly, **where the traditional baseline is already enough.**

---

## The before/after story

**BEFORE — the GIS-analyst index pipeline (`before/rf_indices.py`).**
The classic move: collapse every patch to a handful of physically-motivated
spectral indices — NDVI (vegetation), NDWI (open water), mean SWIR (soil/moisture),
overall brightness — and feed those four numbers to a tabular Random Forest. It is
transparent, CPU-only, and on well-separated classes it is *genuinely hard to beat*.
The honest catch: it throws away all spatial **texture** (it averages each patch to
four scalars), needs a human to choose the right indices per problem, and a
production version (atmospheric correction, cloud masking, index selection,
threshold tuning, QA) is the better part of a **GIS analyst's day or more**.

**AFTER — a small CNN on the raw bands (`after/cnn_classifier.py`).**
Claude Code scaffolds a compact 3-block PyTorch CNN that learns directly from the
raw `(5, H, W)` reflectance cube — capturing texture the indices discard — with
AdamW, a validation split, and **early stopping**. It runs on **CPU for tests** and
**auto-uses CUDA** when a GPU is present (the repo box has an RTX 4090). The same
architecture (global-average-pooled head) transfers from 8×8 test patches to larger
real Sentinel-2 chips with no code change. End-to-end from idea to trained-and-
plotted model: **minutes.**

**AFTER (optional, documented) — pretrained-encoder embeddings + kNN.**
`embedding_knn_baseline()` documents the modern foundation-model route: push each
patch through a *pretrained* remote-sensing encoder (e.g. SatMAE / Prithvi / Clay,
or the vision tower of an RS-MLLM in the spirit of EagleVision, arXiv:2503.23330),
then classify the embeddings with a cheap kNN — competitive **without task-specific
training**. It is intentionally skippable: with **no encoder supplied it returns a
documented skip, never a fabricated number** (runs-anywhere discipline).

**Change detection (`after/change_detect.py`).**
The most interpretable detector in the toolbox: compute NDVI at t0 and t1, take the
per-patch NDVI **drop**, and threshold it. For deforestation NDVI falls sharply, so
a single calibrated threshold already separates changed from unchanged. We report
accuracy / macro-F1 against the ground-truth `changed` labels and render a change
map for one pair.

### An honest note on the synthetic numbers

The committed `results/` come from the deterministic **synthetic** generator in
`common/`, whose class spectral signatures are deliberately well-separated. On that
data **both** the RF-on-indices baseline **and** the fully-trained CNN reach near-
perfect accuracy — the cheap baseline is already excellent, so the CNN's value here
is **capability and scaling**, not a headline accuracy bump:

- the CNN learns from **raw pixels/texture**, so it keeps working when hand-chosen
  indices stop being sufficient (mixed pixels, spectrally-similar classes, urban
  texture) — exactly the regime of real Sentinel-2 scenes;
- it scales to large chips and big tile stacks on the 4090;
- and Claude Code wrote, tested, and plotted the whole pipeline in minutes.

We **do not** claim a synthetic accuracy win the data does not support. The
before/after contrast that *is* real and verifiable: **analyst-hours of bespoke
index engineering → minutes of agent-scaffolded, unit-tested, GPU-ready code.**

---

## Results

Headline metrics live in [`results/metrics.json`](results/metrics.json), with a
readable table in [`results/summary.md`](results/summary.md). Plots:

| Artifact | What it shows |
|---|---|
| `results/confusion_matrix.png` | CNN confusion matrix over the 5 land-cover classes |
| `results/change_map.png` | one pair: BEFORE/AFTER RGB, NDVI-drop heatmap, predicted-vs-true change mask |
| `results/before_after_bars.png` | RF-on-indices vs CNN-on-bands accuracy & macro-F1 |

The committed numbers are produced by a fast `--quick` (CPU) run for
reproducibility-anywhere; the headline run (more data, more epochs, CUDA) is
produced on the 4090 — see *Scaling on a 4090* below.

### Before/after at a glance

| | BEFORE — RF on indices | AFTER — CNN on raw bands |
|---|---|---|
| **Inputs** | 4 hand-computed scalars/patch (NDVI, NDWI, SWIR, brightness) | full `(5,H,W)` reflectance cube |
| **Captures texture?** | no (spatially averaged away) | yes (convolutions) |
| **Hardware** | laptop CPU | CPU for tests; **auto-CUDA** (RTX 4090) for scale |
| **Real-world ceiling** | strong on separable classes; brittle on mixed/urban texture | learns features → higher ceiling on hard scenes; scales to big tiles |
| **Human effort** | GIS analyst: **hours–a day** of index choice + threshold/QA tuning | Claude Code: **minutes** to scaffold, unit-test, train, plot |

---

## How to run

Always run from the **repo root** so `import common` resolves.

```bash
# Fast smoke run (CPU, tiny, ~seconds) — what the committed results use
python experiments/04_remote_sensing_landcover/run_before_after.py --quick --cpu

# Modest default run (~1–2 min on CPU)
python experiments/04_remote_sensing_landcover/run_before_after.py

# Headline-style run, auto-using the GPU if present
python experiments/04_remote_sensing_landcover/run_before_after.py --epochs 60 --n-train 4000 --n-test 1000

# Fast unit tests (CPU, tiny n, 1 epoch)
python -m pytest experiments/04_remote_sensing_landcover/tests -q
```

Flags: `--epochs N`, `--quick` (tiny+fast), `--cpu` (skip CUDA auto-use),
`--n-train/--n-test/--n-change`, `--size`, `--seed`, `--rf-estimators`,
`--results-dir`.

Run individual pieces:

```bash
python experiments/04_remote_sensing_landcover/before/rf_indices.py     # RF baseline
python experiments/04_remote_sensing_landcover/after/cnn_classifier.py  # CNN
python experiments/04_remote_sensing_landcover/after/change_detect.py   # change detection
```

---

## Swapping in real data (Sentinel-2)

The synthetic generator returns exactly the array contract real data uses, so the
swap is a drop-in:

- **Bands.** This experiment uses `[blue, green, red, nir, swir]`, which map to
  **Sentinel-2 L2A bands `B2, B3, B4, B8, B11`**. (B2/B3/B4 are 10 m, B8 is 10 m,
  B11 is 20 m — resample B11 to 10 m so all five bands share a grid.)
- **Loading.** Read the GeoTIFFs with **`rasterio`**, or pull analysis-ready chips
  via **`sentinelhub`** (or `openeo` / the AWS/GCP Sentinel-2 COG buckets). Stack to
  a `(5, H, W)` array, scale L2A reflectance to `[0, 1]` (divide by 10000), and tile
  into patches.
- **Contract to preserve.** Keep `X` as `float32 (N, 5, H, W)` reflectance in `[0,1]`
  and `y` as integer labels into a `CLASSES` list. Replace `multispectral_patches`
  with your loader and **everything downstream (RF, CNN, change detection, plots)
  works unchanged.** For change detection, supply co-registered t0/t1 chips and
  ground-truth change masks (e.g. from a deforestation product); the NDVI-difference
  threshold is then calibrated on a small labelled validation set.
- **Labels.** Use an existing land-cover product (ESA WorldCover, Dynamic World,
  CORINE) or hand-labelled tiles for training/eval.

These dependencies are intentionally **optional** — they are commented out in the
repo's `requirements.txt`; the committed results need none of them.

---

## Scaling on a 4090

The architecture and loop are unchanged from the CPU smoke test to the GPU headline
run — only the data size, patch size, epochs, and batch size grow:

- **Auto-device.** `pick_device()` selects CUDA automatically when available; the
  run script prints the device it used and records it in `metrics.json`.
- **What to scale.** More tiles (`--n-train` into the thousands), larger chips
  (`--size 64`+), more epochs with early stopping doing the right thing, and a
  larger `width`/batch size — comfortably within the 4090's ~16 GB.
- **Why it matters on real scenes.** Texture and context are where a CNN pulls ahead
  of index-only baselines: spectrally-similar classes (bare soil vs harvested
  cropland), mixed pixels, and urban structure. That gap is invisible on the clean
  synthetic signatures here but is the entire point of going to raw-band DL on real
  Sentinel-2 data.
- **Foundation-model route.** For the embeddings+kNN path, load a pretrained RS
  encoder on the GPU and pass an `encoder` callable to `embedding_knn_baseline()` —
  no from-scratch training needed (see the docstring for SatMAE/Prithvi/Clay wiring).

---

## Files

```
04_remote_sensing_landcover/
├── before/rf_indices.py        # RandomForest on hand-computed spectral indices
├── after/cnn_classifier.py     # SmallCNN on raw 5-band patches (+ embedding_knn_baseline)
├── after/change_detect.py      # NDVI-difference change detector
├── run_before_after.py         # orchestrator: metrics.json + 3 PNGs + summary.md
├── tests/test_exp04.py         # fast CPU pytest (<30s compute)
├── results/                    # committed metrics + plots
└── README.md                   # this file
```

*Generator + utilities are committed in `common/` (do not modify): `multispectral_patches`,
`compute_indices`, `change_pair`, `BANDS`, `CLASSES`, `metrics`, `plotting`.*
