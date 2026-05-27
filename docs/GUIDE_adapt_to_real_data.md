# Adapting the experiments to real data

This guide explains, **for each experiment**, how to swap the deterministic synthetic
generator for the real-world data source it stands in for — while keeping the
experiment's code unchanged. Every experiment was written against a fixed **data
contract** (a schema: column names, array shapes, dtypes). The synthetic generator and
the real loader produce the *same* contract, so the swap is a drop-in: replace the
generator call with your loader and `run_before` / `run_after` / the metrics / the plots
all keep working.

> **Why synthetic first?** The headline results in [`RESULTS.md`](../RESULTS.md) are
> reproducible with **no API keys, no network, no GPU** (deterministic seeds). That is a
> deliberate "runs-anywhere" rule — anyone can sanity-check the *pipeline logic* before
> committing to the (real, sometimes large, sometimes key-gated) effort of fetching real
> data. Synthetic numbers are **ceilings/illustrations**, not real-world performance.
> See [`ADDENDUM_methodology.md`](ADDENDUM_methodology.md) for why each generator is a
> fair stand-in and what its limitations are.

## Before you start: shared tooling

- **The `/climate-data-fetch` skill** ([`skills/climate-data-fetch/SKILL.md`](../skills/climate-data-fetch/SKILL.md))
  is the single place that consolidates fetch snippets, install commands, auth notes,
  caching, and provenance for ERA5 / GBIF / Sentinel-2 / OpenAQ / CAMELS. It asks the
  decision question first ("do you need REAL data, or just a stand-in to prototype?") and
  points each dataset back at its synthetic generator. Use it as the authoritative source
  for any fetch; this guide focuses on the per-experiment data contract and swap mechanics.
- **The `/env-stats-validate` skill** ([`skills/env-stats-validate/SKILL.md`](../skills/env-stats-validate/SKILL.md))
  is the rigor checklist to run on any result the real data produces — leakage (temporal
  AND spatial), stationarity, distributional assumptions, multiple testing, uncertainty,
  baselines/skill, reproducibility. Real data fails these checks in ways synthetic data
  cannot; run the checklist before you trust a real-data number.
- **Optional dependencies.** The real-data packages (`cdsapi`, `xarray`, `rasterio`,
  `sentinelhub`, `pygbif`, `anthropic`, `chronos-forecasting`) are intentionally
  **commented out** in [`requirements.txt`](../requirements.txt) under the "Optional —
  real data / foundation model variants" block. The committed results need none of them.
  Install only the ones your chosen swap needs.

## Provenance discipline (do this for every real fetch)

Whatever you fetch, write a sidecar provenance record next to the cached file, as the
`climate-data-fetch` skill specifies:

```
<cache_dir>/<name>.meta.json
{ "source": "...", "query": {...}, "variables": [...], "spatial_extent": [...],
  "time_range": ["...","..."], "fetched_at": "<ISO timestamp>", "tool_version": "...",
  "license": "(verify the dataset's license/terms)", "checksum": "<sha256 of file>" }
```

Cache on a hash of the full query (variables + bbox + dates + resolution); reuse if the
file and `.meta.json` exist and the checksum matches; never silently overwrite a cached
file with a different query. Record dataset **biases** (GBIF sampling bias, Sentinel-2
cloud cover, OpenAQ station gaps) before any modelling.

---

## Exp 01 — Climate time-series forecasting → ERA5 (cdsapi / xarray)

**Source dir:** [`experiments/01_climate_timeseries_forecast/`](../experiments/01_climate_timeseries_forecast/)
**Synthetic generator:** `common.synthetic_climate.daily_temperature`

### Data contract the experiment expects

A `pandas.DataFrame` indexed by a `date` `DatetimeIndex` (one row per day) with a single
column `t2m` (2 m air temperature in °C). That frame is fed to
`common.synthetic_climate.time_split(...)` → `run_before` / `run_after`. Keep this schema
and persistence, seasonal-naive, (S)ARIMA/AutoReg, and the LSTM all run unchanged.

### Swap-in (single-cell / station series)

```bash
pip install cdsapi xarray netCDF4
```

1. Register at the Copernicus **Climate Data Store** and place your CDS API key in
   `~/.cdsapirc`; accept the dataset licence once in the CDS UI. **(verify the current
   CDS endpoint/registration flow — the CDS has migrated services before.)**
2. Download `2m_temperature` (`reanalysis-era5-single-levels`) for a grid cell / region
   and period (see the `climate-data-fetch` ERA5 snippet).
3. Reduce to a daily `t2m` frame:

```python
import xarray as xr
ds = xr.open_dataset("era5_t2m.nc")
t2m = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")
daily = (t2m.resample(time="1D").mean() - 273.15)   # K -> °C
df = daily.to_dataframe(name="t2m")[["t2m"]].rename_axis("date")
```

For **gridded** forecasting (not a single cell), use latitude-weighted RMSE / ACC
(`common.metrics.latitude_weighted_rmse`, `anomaly_correlation`) — the ERA5-benchmark
convention; the synthetic side already exposes a `gridded_temperature_field()` with a
`lats` array for exactly this.

### Effort & keys, honestly

- **Key required** (CDS account + licence acceptance). Requests **queue server-side**;
  the first pull of a large window can take minutes to hours — always cache the `.nc`.
- The full ERA5 archive is petabytes; **for a 4090 / laptop**, pull a small subset (few
  variables, coarse grid, short window). Single-variable monthly subsets are MB–GB.
- **Caching/provenance:** never re-request a window you already have; write the
  `.meta.json` sidecar.
- **Foundation-model row:** the optional `zero_shot_foundation_baseline()` (Chronos /
  TimesFM) is unchanged by the data swap — it consumes the same `t2m` series.

---

## Exp 02 — Climate extremes & trend detection → ERA5 / station (cdsapi / xarray)

**Source dir:** [`experiments/02_extreme_value_trends/`](../experiments/02_extreme_value_trends/)
**Synthetic generator:** `common.synthetic_climate.daily_precipitation`

### Data contract the experiment expects

A `DataFrame` with a `date` index plus a `precip` column (daily precipitation, mm).
`annual_maxima`, `rx5day`, `r95p`, `mann_kendall`, `fit_gev`,
`bootstrap_return_levels`, and `validate` are all data-agnostic given this schema.

### Swap-in

```bash
pip install cdsapi xarray netCDF4
```

1. **ERA5:** download `total_precipitation` from the CDS (same auth as Exp01), aggregate
   to a daily total at your grid cell / region with xarray, rename to `precip`.
2. **Station data:** any daily-precip table works — e.g. **GHCN-Daily** `PRCP`. Reduce to
   the `date` + `precip` schema.

### Effort & keys, honestly

- ERA5 path needs the CDS key (as Exp01). The station path (GHCN-Daily) is a **free,
  keyless** download — often the lower-effort route for a single record.
- **Rigor caveat the `validate()` gate already surfaces:** real warming-climate records
  are typically **non-stationary** (a trend in the maxima), which violates the
  stationary-GEV assumption — the gate warns and points to a non-stationary GEV with
  time-varying location/scale. Also check serial dependence in the blocks and record
  length (return levels far beyond the record are extrapolations — report the CI).

---

## Exp 03 — Biodiversity species-interaction extraction → iNaturalist / GBIF text

**Source dir:** [`experiments/03_biodiversity_text_extraction/`](../experiments/03_biodiversity_text_extraction/)
**Synthetic generator:** `common.synthetic_biodiversity.citizen_comments` (+ `gold_interactions`)

### Data contract the experiment expects

A list of records with the schema `{"id", "text", "interactions"}`, where `interactions`
is a list of typed triples `(subject, relation, object)` over the closed relation
vocabulary `pollinates | feeds_on | parasitizes | competes_with | depends_on`. Leave
`interactions` empty (`[]`) for unlabeled data; fill it only for a held-out gold set. The
scorer and graph builder are data-agnostic.

### Swap-in

```bash
pip install pygbif      # for GBIF; iNaturalist uses its REST API (requests)
```

1. Pull observation **comments / descriptions / field notes** from **iNaturalist**
   (per-observation `description` and comment fields) or occurrence remarks from **GBIF**
   (`occurrenceRemarks`, `eventRemarks`).
2. Map each record to `{"id": <obs id>, "text": <comment>, "interactions": []}`.
3. Point `run_before_after.py` at your loader instead of `citizen_comments(...)`.
4. Replace `SPECIES` / `RELATIONS` with a domain taxonomy (e.g. a GBIF backbone name list).

### Effort & keys, honestly

- Small GBIF `occurrences.search` queries need **no key**; large **download** requests
  need a free GBIF account. iNaturalist has an API + open data on AWS; metadata is light.
  **(verify current API versions/limits.)**
- The real **LLM extractor** (`extract_with_llm`) is gated behind `ANTHROPIC_API_KEY` and
  needs `pip install anthropic`. Without the key the deterministic offline `extract()`
  (the behavioural stand-in) runs — that is what the tests and committed results use.
- **The honest catch:** the synthetic 1.00 P/R/F1 is a **ceiling on a closed-vocabulary
  corpus**, *not* a real-world number. On real text expect lower, noisier scores. Extracted
  triples are a **hypothesis set, not ground truth** — a human ecologist must verify
  relation type/direction before any triple enters a food-web/network analysis. This is
  the mandatory human-in-the-loop step, not optional.

---

## Exp 04 — Remote-sensing land cover & change → Sentinel-2 (rasterio / sentinelhub)

**Source dir:** [`experiments/04_remote_sensing_landcover/`](../experiments/04_remote_sensing_landcover/)
**Synthetic generator:** `common.synthetic_remote_sensing.multispectral_patches`
(+ `multispectral_patches_hard`, `compute_indices`, `change_pair`)

### Data contract the experiment expects

- `X`: `float32` array `(N, 5, H, W)` of surface reflectance scaled to `[0, 1]`, bands
  ordered `[blue, green, red, nir, swir]`.
- `y`: integer labels indexing into `CLASSES` (`water, forest, cropland, urban, bare`).
- Change detection: co-registered `t0` / `t1` chips + a ground-truth `changed` mask.

Keep this contract and the RF-on-indices baseline, the CNN, change detection, and all
plots run unchanged.

### Swap-in

```bash
pip install sentinelhub rasterio      # or: pip install pystac-client rasterio
```

- **Bands.** Map the 5 bands to **Sentinel-2 L2A `B2, B3, B4, B8, B11`** (blue, green,
  red, NIR, SWIR). B2/B3/B4/B8 are 10 m; **B11 is 20 m — resample it to 10 m** so all
  five share a grid.
- **Loading.** Read GeoTIFFs with **`rasterio`** (windowed reads of cloud-optimized
  GeoTIFFs avoid whole-scene downloads), or pull analysis-ready chips via **`sentinelhub`**
  / a STAC client (Earth Search / Planetary Computer). Stack to `(5, H, W)`, **scale L2A
  reflectance to `[0,1]` by dividing by 10000**, and tile into patches.
- **Labels.** Use an existing land-cover product (ESA WorldCover, Dynamic World, CORINE)
  or hand-labelled tiles. For change detection, supply co-registered t0/t1 chips and a
  ground-truth change mask (e.g. a deforestation product); calibrate the NDVI-difference
  threshold on a small labelled validation set.

### Effort & keys, honestly

- **Sentinel Hub** needs OAuth client credentials (client id/secret via `SHConfig`).
  Open **STAC** catalogs (Earth Search / Planetary Computer) may need **no key** but have
  their own terms. **(verify current auth flow and catalog URLs/asset keys — they change.)**
- Tiles are ~100s MB–GB each; **work at single-tile / single-AOI scale** and downsample
  for 4090 vision demos. Real Sentinel-2 also brings **cloud cover** (filter on
  `eo:cloud_cover`) and atmospheric/QA handling that synthetic data lacks.
- **On the synthetic numbers:** the default classes are well-separated, so RF-on-indices
  already near-ties the CNN — the win there is *effort/capability*, not accuracy. The
  **hard-mode** variant (`multispectral_patches_hard`, texture-only classes) is where the
  CNN genuinely wins (+0.356 accuracy in the committed 4090 run). Real Sentinel-2 scenes
  (mixed pixels, urban texture, spectrally-similar classes) are the regime where the CNN's
  texture advantage matters.

---

## Exp 06 — Spatial interpolation → station networks + DEM (SRTM)

**Source dir:** [`experiments/06_spatial_interpolation/`](../experiments/06_spatial_interpolation/)
**Synthetic generator:** `common.synthetic_spatial.synthetic_spatial_field`

### Data contract the experiment expects

A `SpatialDataset` with train arrays (`train_coords[n,2]`, `train_covariate[n]`,
`train_value[n]`) and a dense scoring grid (`grid_coords[m,2]`, `grid_covariate[m]`,
`grid_value[m]`, `grid_shape`). Keep these names/shapes and IDW, ordinary kriging (GP),
the empirical variogram, and the covariate-aware RF/GBM all run unchanged.

### Swap-in

1. **Stations** — observed 2 m temperature from **GHCN-Daily** or a national met service
   (or a pollutant from **OpenAQ**): build `train_coords` and `train_value`. Use a
   **projected CRS in km**, not raw lon/lat, so distances are Euclidean.
2. **Covariate** — sample **elevation** from a **DEM** (**SRTM**, or the Copernicus DEM)
   at each station (`train_covariate`) and resample the DEM to your prediction grid
   (`grid_covariate`). Tools: `rasterio` to read the DEM raster; sample at station points;
   resample to the target grid. **(verify the DEM product/host and its licence.)**
3. **Scoring grid** — `grid_coords` over the target raster; `grid_value` from a held-out
   station subset (or a trusted reanalysis) for honest evaluation.

### Effort & keys, honestly

- GHCN-Daily and SRTM/Copernicus DEM are **free, keyless** downloads — relatively low
  effort. OpenAQ may need a free key (see Exp07). The real work is **CRS alignment** and
  **resampling the DEM to the station points and the grid** — easy to get subtly wrong
  (off-by-a-pixel, wrong CRS) and a classic source of silent error.
- **The uncertainty tradeoff is real, not a strawman.** The covariate-aware tree model
  wins on point RMSE but gives **no calibrated variance surface**; ordinary kriging does.
  The genuinely "hero" workflow combines them — **regression-kriging** (krige the residuals
  of a covariate regression) or a **GP whose mean/inputs include the covariate** — for both
  lower error and a calibrated uncertainty map. Plug the DEM covariate into that step.

---

## Exp 07 — Air-quality PM2.5 nowcast → OpenAQ + reanalysis meteorology

**Source dir:** [`experiments/07_air_quality_nowcast/`](../experiments/07_air_quality_nowcast/)
**Synthetic generator:** `common.synthetic_airquality.synthetic_pm25`

### Data contract the experiment expects

A `DataFrame` indexed by `date` with columns
`['pm25', 'wind', 'temp', 'boundary_layer', 'hour', 'dow', 'is_weekend']`, resampled to a
common hourly (or daily) index. Keep this schema and persistence / linear-AR / ARIMA and
the covariate GBM/MLP all run unchanged.

### Swap-in

1. **PM2.5** — pull a station's series from **OpenAQ** (the v3 REST API, or the open S3
   archive). **(verify the current OpenAQ API base URL/version and whether an API key
   header is required.)**
2. **Meteorology** — get matching **10 m wind speed**, **2 m temperature**, and
   **planetary boundary-layer height** from a reanalysis such as **ERA5** (CDS via
   `cdsapi`/`xarray`) or a co-located met station. ERA5's boundary-layer height is the
   standard ventilation/dilution (inversion) proxy.
3. Resample everything to a common index, assemble the DataFrame with the **same columns**,
   feed it into `time_split(...)` → `run_before` / `run_after`.
4. For multi-station models, concatenate stations and add static site attributes (land
   use, elevation) as extra columns — the feature matrix generalizes with no code changes.

### Effort & keys, honestly

- OpenAQ is free; current API versions **may require a free API key** — **(verify)**.
  ERA5 needs the CDS key + licence (as Exp01). So this experiment couples a (probably
  keyed) air-quality fetch with a (definitely keyed, queued) reanalysis fetch — the
  meteorology side is the bigger lift.
- **QA/QC the PM2.5:** drop negatives, flag sensor drift, handle station gaps — real
  air-quality series are messy. **No weather feature may include the PM2.5 target** (no
  target leakage); the nowcast uses *contemporaneous* meteorology because met/reanalysis
  report in real time while the reference PM2.5 monitor is sparse/delayed.
- **The lesson is "bring in the physical drivers", not "use a fancier model":** the
  committed ablation shows the same GBM with weather removed is only modestly better than
  persistence. The win comes from the covariates; an agentic workflow makes pulling and
  aligning them a minutes-long task.

---

## Exp 08 — Hydrology rainfall-runoff streamflow → CAMELS

**Source dir:** [`experiments/08_hydrology_streamflow/`](../experiments/08_hydrology_streamflow/)
**Synthetic generator:** `common.synthetic_hydrology.synthetic_catchment`

### Data contract the experiment expects

A `DataFrame` indexed by `date` (one row per day) with columns
`['precip', 'temperature', 'pet', 'streamflow']`. Keep this schema and the linear
baseline, the conceptual bucket, and the LSTM all run unchanged.

### Swap-in

```bash
pip install pandas      # CAMELS files are CSV/text — no special API client
```

1. Download a basin's daily forcings (precipitation, temperature, PET) and observed
   discharge from a **CAMELS** release — US CAMELS (Newman et al. / Addor et al.) or a
   regional twin (**CAMELS-GB, -CL, -BR, -AUS**). **(verify the current host/DOI and
   licence for the variant you use.)**
2. Assemble a `DataFrame` with the **same columns** above, indexed by a daily
   `DatetimeIndex`.
3. Feed it into `time_split(...)` → `run_before` / `run_after`.
4. For multi-basin training (where LSTMs shine most), concatenate basins and add static
   catchment attributes as extra input features — the LSTM input tensor
   `(batch, lookback, n_features)` generalizes with no model code changes.

### Effort & keys, honestly

- **No API key** — CAMELS is a free static download, **small enough to fit fully in RAM**.
  This is the **lowest-friction real-data swap** in the repo (just a download + a schema
  map). It is also the cleanest "AFTER wins" example: LSTM rainfall-runoff matching/exceeding
  calibrated conceptual baselines is a well-established large-sample result.
- **Leakage discipline:** **split by basin (spatial group)** so basins never cross
  train/test; never shuffle the within-basin time series. Run `env-stats-validate` #1.

---

## Exp 05 — AutoResearch climate loop → ERA5 (the flagship)

**Source dir:** [`experiments/05_autoresearch_climate/`](../experiments/05_autoresearch_climate/)
**Synthetic generator:** `common.synthetic_climate.daily_temperature`
(via the loop's `load_dataset`)

The flagship loop drives a gated, citation-grounded sequence of forecasting experiments.
Its data swap is **the same ERA5 swap as Exp01** — point `load_dataset` at real ERA5
(see `common.synthetic_climate.REAL_DATA_NOTE`) or at many station-years. The
walk-forward-by-year and super-fold-by-regime splits carry over directly to a real,
non-stationary, warming series. The **gates, composite metric, frozen fingerprint,
logging, and checkpoint are untouched by the data swap** — only the backbone and data
grow (e.g. swap the tiny numpy MLP for a torch LSTM or a zero-shot TS foundation model on
the 4090). See the experiment README's "Scaling up on the 4090" section and
[`docs/autoresearch_protocol.md`](autoresearch_protocol.md).

---

## Quick reference

| Exp | Real source | Key/auth required? | Friction | Primary contract |
|---|---|---|---|---|
| 01 | ERA5 (`cdsapi`/`xarray`) | CDS key + licence | Medium (queued, large) | `date`-indexed `t2m` (°C) frame |
| 02 | ERA5 or GHCN-Daily | CDS key (ERA5) / none (GHCN) | Low–Medium | `date` + `precip` (mm) frame |
| 03 | iNaturalist / GBIF text | none for search; key for LLM | Low (but human-verify) | `{id, text, interactions[]}` records |
| 04 | Sentinel-2 (`rasterio`/`sentinelhub`) | OAuth (Sentinel Hub) / none (STAC) | High (tiles, clouds, CRS) | `X (N,5,H,W)` in [0,1], int `y` |
| 05 | ERA5 (flagship loop) | CDS key + licence | Medium | same as Exp01 |
| 06 | GHCN/OpenAQ + DEM (SRTM) | none (GHCN/SRTM) / maybe (OpenAQ) | Medium (CRS/resample) | `SpatialDataset` coords/covariate/value |
| 07 | OpenAQ + ERA5 met | maybe (OpenAQ) + CDS key | High (two keyed fetches) | `date` + pm25/wind/temp/BLH/calendar |
| 08 | CAMELS | none | **Low** (static download) | `date` + precip/temperature/pet/streamflow |

For every row: write a `.meta.json` provenance sidecar, key the cache on the full query,
note dataset biases, and run `env-stats-validate` before trusting a number.
