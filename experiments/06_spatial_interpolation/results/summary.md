# Experiment 06 — Spatial interpolation: BEFORE vs AFTER

- Device: `cpu`  |  stations: 180  |  held-out grid: 45×45 = 2025 cells
- AFTER model: RF (n_estimators=400)  |  quick: False  |  seed: 0
- Covariate informativeness on the grid: corr(value, covariate) = -0.939, univariate R² = 0.882

## Interpolation error (held-out dense grid, noise-free truth)

| Method | Uses covariate? | RMSE ↓ | MAE ↓ |
|--------|:--------------:|------:|------:|
| Before: IDW (coords only) | no | 4.936 | 3.642 |
| Before: ordinary kriging / GP (coords only) | no | 2.905 | 2.237 |
| **After: RF + covariate** | **yes** | **2.203** | **1.599** |
| _AFTER − kriging_ | | -0.702 | -0.638 |

AFTER reduces RMSE vs coordinate-only kriging by 24.2% (skill score 1 − RMSE_ml/RMSE_krig = 0.242). Top AFTER features: covariate (0.57), x*cov (0.26), covariate^2 (0.08).

## Honest tradeoff — uncertainty surface

Kriging supplies a calibrated predictive-variance surface (mean predictive std ~2.75); the vanilla ML model gives a lower-RMSE point map but no such uncertainty surface. Lower RMSE is not the whole story.

Ordinary kriging is not merely 'the old way': it yields a principled, spatially-varying **predictive-variance** map (largest far from stations) that is exactly what many environmental deliverables require (risk maps, network design, data-assimilation weights). The covariate-aware ML model here wins on point accuracy but does **not** provide that calibrated uncertainty for free. The genuinely 'hero' workflow combines them: use the covariate (regression-kriging / GP with the covariate as an input or mean function) to get BOTH lower error AND an uncertainty surface.

## Wall time (this machine)

- BEFORE (IDW + kriging fit): 5.43 s   |   AFTER (ML fit + predict): 2.68 s

Artifacts: `predicted_surface.png`, `variogram.png`, `before_after_bars.png`, `metrics.json`.

Swap in real data: To use real data instead of this synthetic field, swap in a station network and a covariate raster: e.g. observed 2 m temperature from GHCN-Daily / a national met service (or a pollutant from OpenAQ), and ELEVATION sampled from a DEM (SRTM or the Copernicus DEM) at each station and at every prediction cell. Assemble: (1) train arrays coords[n,2], covariate[n], value[n]; (2) a dense grid grid_coords[m,2], grid_covariate[m] (DEM resampled to the target grid) and, for scoring, grid_value[m] from a held-out station subset or a trusted reanalysis. Keep these names/shapes and IDW, kriging, the covariate-aware ML, the runner and the tests all work unchanged. (Standardise coords to a projected CRS in km and scale the covariate as done here.)
