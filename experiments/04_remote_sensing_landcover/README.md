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

## Hard mode — where the CNN *genuinely* beats RF-on-indices

The default synthetic classes have well-separated mean spectra, so RF-on-indices is
already near-perfect and the CNN can only tie it — the honest before/after story
there is *effort*, not *accuracy*. **Hard mode** adds the missing accuracy win by
making the task **texture-separable**.

`common.synthetic_remote_sensing.multispectral_patches_hard(n, size, seed)` builds
five classes in which two **pairs share a (near-)identical mean spectrum** and differ
**only in spatial texture**:

| Class | Base spectrum | Distinguishing spatial pattern |
|---|---|---|
| `forest_natural` | forest | isotropic, low-frequency canopy mottle |
| `orchard_rows` | forest (same NDVI/NDWI) | oriented, periodic planting **rows** |
| `water_smooth` | water | isotropic wind-ripple texture |
| `flooded_field` | water (same spectrum) | oriented **rows** from submerged crop rows |
| `bare` | bare | isotropic mid-frequency texture (spectral anchor) |

The texture is added as a **per-band, zero-mean** spatial field, and within each pair
the two members are matched in amplitude *and* spatial-correlation magnitude (differing
only in isotropy vs orientation). So NDVI / NDWI / mean-SWIR / brightness — everything
`compute_indices()` returns — are **structurally blind** to the pairs, while a CNN reads
the oriented pattern directly. (We deliberately avoided a smooth-vs-white-speckle pair:
that leaks an NDVI-*variance* tell a tree ensemble can exploit. See the module note.)

This is verified by `common/tests/test_remote_sensing_hard.py`: a linear classifier on
`compute_indices()` scores near chance on each texture pair (and the 5-class index
ceiling is well below the texture ceiling), while simple translation-invariant texture
statistics — the kind of signal a CNN learns — separate the same pairs near-perfectly.

### Real result (RTX 4090, `after/cnn_hard.py`)

Same two pipelines, run on the **hard** data (`n_train=3000`, `n_test=1000`, `size=16`,
`seed=0`, RF `n_estimators=300`, CNN `width=24`, 40 epochs with early stopping → stopped
at epoch 32, best epoch 23, 53,741 params, **device=cuda**):

| Pipeline | Features | Accuracy | Macro-F1 |
|---|---|---:|---:|
| BEFORE — RandomForest-on-indices | 4 hand-computed indices | **0.642** | **0.641** |
| AFTER — SmallCNN-on-bands | raw `(5,16,16)` patches | **0.998** | **0.998** |
| **Δ (after − before)** | | **+0.356** | **+0.357** |

**Honest win:** on texture-separable land cover the CNN beats RF-on-indices by
**+0.356 accuracy** and **+0.357 macro-F1**. RF caps at ~0.64 because it cannot resolve
the two same-spectrum texture pairs (its confusion is *within* `forest_natural↔orchard_rows`
and `water_smooth↔flooded_field`); the CNN resolves both. Artifacts in
[`results/hard/`](results/hard/): `metrics.json`, `summary.md`,
`confusion_matrix_hard.png`, `before_after_hard_bars.png`.

```bash
# Hard-mode before/after (auto-CUDA on the 4090, CPU fallback)
python experiments/04_remote_sensing_landcover/after/cnn_hard.py
python experiments/04_remote_sensing_landcover/after/cnn_hard.py --quick --cpu   # fast smoke
```

---

## Foundation encoder — real pretrained ViT, frozen, on HARD mode

Beyond the from-scratch CNN, we also ran a **real pretrained remote-sensing
foundation encoder** — **NASA-IBM Prithvi-EO-1.0-100M** (Hugging Face id
[`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M),
Apache-2.0; ViT-B/16, 100 M params, pretrained with a Masked Autoencoder on
NASA's Harmonized Landsat-Sentinel-2 (HLS) product) — **frozen** as a feature
extractor, with two cheap supervised heads on top:

- a **linear probe** (multinomial `LogisticRegression` on standardised CLS
  embeddings — the canonical "is the embedding linearly separable?" protocol),
- a **kNN(k=5)** head (no training at all, pure nearest-neighbour transfer).

We chose Prithvi over option (1) **EagleVision (arXiv:2503.23330)** — an
RS-**MLLM** built around captioning/QA, with no clean turn-key embedding
endpoint released as a Hugging Face model — and over option (3) a generic
ImageNet `timm` ViT, because Prithvi is the closest *RS-specific* pretrained
encoder for our spectral domain (we feed 5 bands → 6 by repeating SWIR for
SWIR1+SWIR2; bilinearly upsample 16×16 patches to 224×224; scale our [0,1]
reflectance to HLS units ×10000 then apply Prithvi's published per-band mean/std).
The generic ImageNet fallback is implemented (`run_foundation.py --encoder
timm`) and documented; it is **not** the headline number.

### Real result (RTX 4090, `run_foundation.py`)

Same HARD-mode data and seeded train/test split as the from-scratch CNN row
above (`n_train=3000`, `n_test=1000`, `size=16`, `seed=0`, device=cuda, embed
36.9 s + heads 8.1 s):

| Pipeline | Trainable params on this task | Accuracy | Macro-F1 |
|---|---|---:|---:|
| BEFORE — RandomForest on indices | RF only | **0.642** | **0.641** |
| AFTER — from-scratch SmallCNN | 53,741 | **0.998** | **0.998** |
| AFTER — Prithvi-EO-100M (frozen) + linear probe | **0** (encoder frozen) | **0.997** | **0.997** |
| AFTER — Prithvi-EO-100M (frozen) + kNN(k=5)     | **0** (no training) | **0.954** | **0.954** |

Honest verdict — the frozen 100 M-parameter HLS-pretrained encoder, with a tiny
linear probe on top, **statistically matches** the bespoke from-scratch CNN
(−0.001 accuracy, −0.001 macro-F1, well inside noise on a 1 000-patch test set)
and **demolishes** the RF-on-indices baseline (**+0.355 accuracy**, **+0.356
macro-F1**). The pure kNN head also blows past RF (+0.312 / +0.313) without any
gradient steps at all, which is the whole foundation-model promise: useful
features out-of-the-box. Caveats we explicitly preserve: (i) this is *synthetic
texture-pair* data, not real HLS scenes, and our generator does not model two
SWIR sub-bands; (ii) the linear-probe and CNN are within 1 patch (out of 1000)
of each other, so the "tie" is the honest read — neither dominates. Full
numbers in [`results/hard/foundation_metrics.json`](results/hard/foundation_metrics.json),
chart in [`results/hard/foundation_comparison.png`](results/hard/foundation_comparison.png).

```bash
# Real foundation encoder + linear/kNN heads on HARD mode (additive only)
python experiments/04_remote_sensing_landcover/run_foundation.py            # GPU/CPU auto
python experiments/04_remote_sensing_landcover/run_foundation.py --quick    # fast smoke
python experiments/04_remote_sensing_landcover/run_foundation.py --encoder timm  # generic-ImageNet fallback
```

---

## Files

```
04_remote_sensing_landcover/
├── before/rf_indices.py             # RandomForest on hand-computed spectral indices
├── after/cnn_classifier.py          # SmallCNN on raw 5-band patches (+ embedding_knn_baseline)
├── after/change_detect.py           # NDVI-difference change detector
├── after/cnn_hard.py                # HARD-mode orchestrator: RF-indices vs CNN-bands (honest win)
├── after/foundation_encoder.py      # NASA-IBM Prithvi-EO-100M (frozen) wrapper + timm fallback
├── run_before_after.py              # orchestrator: metrics.json + 3 PNGs + summary.md
├── run_foundation.py                # ADDITIVE: foundation_metrics.json + foundation_comparison.png
├── tests/test_exp04.py              # fast CPU pytest (<30s compute)
├── tests/test_exp04_hard.py         # fast CPU pytest for the hard path
├── tests/test_foundation_imports.py # fast offline test of the Prithvi wrapper plumbing
├── results/                         # committed metrics + plots (default mode)
├── results/hard/                    # committed metrics + plots (hard mode, real 4090 run)
│   ├── metrics.json                 # RF-indices vs from-scratch CNN (existing, untouched)
│   ├── foundation_metrics.json      # Prithvi (frozen) + linear/kNN (NEW, additive)
│   └── foundation_comparison.png    # 3-way bar chart: RF vs CNN vs frozen-Prithvi heads
└── README.md                        # this file
```

*Generator + utilities are committed in `common/` (do not modify): `multispectral_patches`,
`multispectral_patches_hard`, `compute_indices`, `change_pair`, `BANDS`, `CLASSES`,
`HARD_CLASSES`, `metrics`, `plotting`.*
