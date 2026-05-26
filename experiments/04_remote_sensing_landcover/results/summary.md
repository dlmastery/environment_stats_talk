# Experiment 04 — Land-cover classification + change detection

_Config: device=cuda, epochs=40, quick=False, n_train=1500, n_test=500, seed=0._

Classes: water, forest, cropland, urban, bare.  Bands: blue, green, red, nir, swir.

## Land-cover classification (same seeded train/test split)

| Pipeline | Features | Accuracy | Macro-F1 |
|---|---|---:|---:|
| BEFORE — RandomForest-on-indices | 4 hand-computed indices | 1.000 | 1.000 |
| AFTER — SmallCNN-on-bands | raw 5-band patches (24,309 params) | 1.000 | 1.000 |
| **Δ (after − before)** | | **+0.000** | **+0.000** |

## Change detection (NDVI-difference threshold)

- threshold (calibrated for max macro-F1): **0.095**
- accuracy: **1.000**, macro-F1: **1.000** (n=600)

## Artifacts

- `confusion_matrix.png` — CNN confusion matrix
- `change_map.png` — before/after + NDVI-drop + change mask
- `before_after_bars.png` — RF vs CNN accuracy & macro-F1

_Note: numbers above are from whatever run produced them (e.g. `--quick`). The headline run (more data, more epochs, CUDA) is produced on the 4090._