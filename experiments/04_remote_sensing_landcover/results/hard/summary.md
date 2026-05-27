# Experiment 04 — HARD mode (texture-separable land cover)

_Config: device=cuda, epochs=40, quick=False, n_train=3000, n_test=1000, size=16, seed=0, rf_estimators=300, cnn_width=24._

Classes: forest_natural, orchard_rows, water_smooth, flooded_field, bare.  Bands: blue, green, red, nir, swir.

Several classes share a (near-)identical **mean spectrum** and differ ONLY in spatial **texture** (natural forest vs orchard rows; open water vs flooded crop rows). Mean indices (NDVI/NDWI/SWIR/brightness) are blind to those pairs, so RF-on-indices is capped; the CNN reads the texture and wins on accuracy.

## Land-cover classification (same seeded train/test split)

| Pipeline | Features | Accuracy | Macro-F1 |
|---|---|---:|---:|
| BEFORE — RandomForest-on-indices | 4 hand-computed indices | 0.642 | 0.641 |
| AFTER — SmallCNN-on-bands | raw 5-band patches (53,741 params) | 0.998 | 0.998 |
| **Δ (after − before)** | | **+0.356** | **+0.357** |

**Honest win:** the CNN beats RF-on-indices by **+0.356 accuracy** and **+0.357 macro-F1** on the texture-separable task.

## Artifacts

- `confusion_matrix_hard.png` — CNN confusion matrix (HARD classes)
- `before_after_hard_bars.png` — RF vs CNN accuracy & macro-F1