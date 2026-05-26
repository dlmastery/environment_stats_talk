---
name: climate-data-fetch
description: >
  Guided retrieval of common environmental datasets with caching and provenance.
  Use when the user says "/climate-data-fetch", "download ERA5 / GBIF / Sentinel-2 /
  OpenAQ / CAMELS", "get me climate/biodiversity/satellite/air-quality data", or needs
  a minimal, reproducible fetch snippet with auth notes and a cache. For each dataset
  it gives: what it is, install, a minimal snippet, the auth/key note, a caching tip,
  and a pointer to this repo's offline synthetic generator for keyless prototyping.
version: 0.1.0
triggers:
  - "/climate-data-fetch"
  - "download ERA5"
  - "fetch GBIF occurrences"
  - "get Sentinel-2 imagery"
  - "pull OpenAQ data"
  - "get CAMELS streamflow"
---

# climate-data-fetch — reproducible environmental-data retrieval

Help the user fetch a real environmental dataset with caching and provenance, OR point
them at this repo's offline synthetic generator when they just want to prototype with
no keys, no network, and deterministic seeds.

**Decision first — ask before fetching anything:**

> "Do you need REAL data (needs an account/API key and network), or just a stand-in to
> prototype the pipeline?"
>
> If prototype → use `common/synthetic_*.py` (no key, no network, deterministic). See
> the "synthetic stand-in" row for each dataset below. Every generator documents how to
> swap in the real dataset later while keeping the same schema, so downstream code is
> unchanged.

**Provenance discipline (audience = scientists).** Whatever you fetch, write a sidecar
provenance record next to the cached file:

```
<cache_dir>/<name>.meta.json
{ "source": "...", "query": {...}, "variables": [...], "spatial_extent": [...],
  "time_range": ["...","..."], "fetched_at": "<ISO timestamp>", "tool_version": "...",
  "license": "(verify the dataset's license/terms)", "checksum": "<sha256 of file>" }
```

**Caching tip (applies to all):** key the cache on a hash of the full query (variables +
bbox + dates + resolution). Skip the download if the cache file AND its `.meta.json`
exist and the checksum matches. Never silently overwrite a cached file with a different
query.

Package names/versions below match this repo's `requirements.txt` (commented "optional —
real-data variants" block). Anything not independently confirmable is marked **(verify)**.

---

## ERA5 — reanalysis climate fields

- **What it is:** ECMWF ERA5 global reanalysis (e.g. `2m_temperature`,
  `total_precipitation`) on a lat/lon grid; the standard backbone for climate forecasting
  and extremes work.
- **Install:** `pip install cdsapi xarray` (repo pins `cdsapi>=0.7`, `xarray>=2024.0`).
- **Auth/key:** register at the Copernicus Climate Data Store and place your CDS API key
  in `~/.cdsapirc` (verify the current CDS endpoint/registration flow — the CDS has
  migrated services before). Accept the dataset licence in the CDS UI once.
- **Minimal snippet:**
  ```python
  import cdsapi, xarray as xr
  c = cdsapi.Client()
  c.retrieve("reanalysis-era5-single-levels", {
      "product_type": "reanalysis", "variable": "2m_temperature",
      "year": "2020", "month": "01", "day": [f"{d:02d}" for d in range(1, 32)],
      "time": ["00:00", "06:00", "12:00", "18:00"],
      "area": [60, -10, 35, 30],   # N, W, S, E
      "format": "netcdf",
  }, "data/era5_t2m_2020-01.nc")
  ds = xr.open_dataset("data/era5_t2m_2020-01.nc")
  ```
- **Caching tip:** ERA5 requests queue server-side and files are large — always cache the
  `.nc` and reuse via `xr.open_dataset`; never re-request a window you already have.
- **Synthetic stand-in:** `common/synthetic_climate.py` →
  `daily_temperature()` / `daily_precipitation()` (point series) and
  `gridded_temperature_field()` (a lat/lon field with `lats` for latitude-weighted RMSE).
  See `REAL_DATA_NOTE` in that file for the exact swap-in.

## GBIF — species occurrence / biodiversity

- **What it is:** Global Biodiversity Information Facility — aggregated species occurrence
  records (presence-heavy, sampling-biased; correct for effort).
- **Install:** `pip install pygbif` (repo pins `pygbif>=0.6`).
- **Auth/key:** small `occurrences.search` queries need no key; large **download**
  requests require a free GBIF account (username/password/email) (verify current limits).
- **Minimal snippet:**
  ```python
  from pygbif import occurrences as occ
  res = occ.search(scientificName="Danaus plexippus", country="MX",
                   hasCoordinate=True, limit=300)
  records = res["results"]  # list of dicts with decimalLatitude/Longitude, eventDate, ...
  ```
- **Caching tip:** cache the raw JSON keyed on the query; record the GBIF `datasetKey`s in
  provenance so the source datasets are citable. Watch for sampling bias before modelling.
- **Synthetic stand-in:** `common/synthetic_biodiversity.py` → `citizen_comments()` (free-
  text observations embedding species interactions) + `gold_interactions()` (gold labels),
  for prototyping text-extraction / interaction-network pipelines without GBIF access.

## Sentinel-2 — multispectral satellite imagery

- **What it is:** ESA Sentinel-2 surface reflectance (bands B2/B3/B4/B8/B11 etc.) for land
  cover, vegetation indices, and change detection.
- **Install:** `pip install sentinelhub rasterio` (repo pins `sentinelhub>=3.10`,
  `rasterio>=1.3`). A STAC client (e.g. `pystac-client`) is a common alternative (verify).
- **Auth/key:** Sentinel Hub needs OAuth client credentials (client id/secret) configured
  via `SHConfig` (verify current auth flow). Open STAC catalogs (e.g. Earth Search /
  Planetary Computer) may need no key but have their own terms (verify).
- **Minimal snippet (STAC-style search, then read a band with rasterio):**
  ```python
  # pip install pystac-client rasterio  (verify catalog URL/terms)
  from pystac_client import Client
  cat = Client.open("https://earth-search.aws.element84.com/v1")  # (verify)
  items = cat.search(collections=["sentinel-2-l2a"],
                     bbox=[-99.3, 19.2, -98.9, 19.6],
                     datetime="2023-01-01/2023-03-31",
                     query={"eo:cloud_cover": {"lt": 20}}).item_collection()
  import rasterio
  href = items[0].assets["red"].href   # asset keys vary by catalog (verify)
  with rasterio.open(href) as src:
      red = src.read(1)
  ```
- **Caching tip:** cache by tile id + date + band; store the `eo:cloud_cover` and scene id
  in provenance. Reading via cloud-optimized GeoTIFF (windowed reads) avoids whole-scene
  downloads.
- **Synthetic stand-in:** `common/synthetic_remote_sensing.py` → `multispectral_patches()`
  (5-band patches + land-cover labels), `compute_indices()` (NDVI/NDWI/...), and
  `change_pair()` (before/after deforestation patches with a change label). Same `(N,C,H,W)`
  reflectance schema as real Sentinel-2 chips, so the swap-in is drop-in.

## OpenAQ — air-quality station data

- **What it is:** Aggregated open air-quality measurements (PM2.5, NO2, O3, ...) from
  ground monitoring stations worldwide; point time series, good for nowcasting/exceedance.
- **Install:** `requests` is enough (HTTP API); a dedicated client may exist (verify).
- **Auth/key:** the OpenAQ API may require a free API key for current versions (verify the
  current API version and whether a key/header is required).
- **Minimal snippet (HTTP — verify base URL/version and key requirement):**
  ```python
  import requests
  # Confirm the current OpenAQ API base URL and auth header before relying on this.
  r = requests.get("https://api.openaq.org/v3/measurements",  # (verify version)
                   params={"parameter": "pm25", "country": "MX", "limit": 1000},
                   headers={"X-API-Key": "<your-key-if-required>"})  # (verify)
  data = r.json()["results"]
  ```
- **Caching tip:** cache per station + parameter + date range; run QA/QC (drop negatives,
  flag drift) and record the station ids and units in provenance.
- **Synthetic stand-in:** `common/synthetic_climate.py` → `daily_precipitation()` or
  `daily_temperature()` give intermittent / persistent station-style series for prototyping
  exceedance / nowcasting pipelines without an OpenAQ key.

## CAMELS — catchment hydrology / streamflow

- **What it is:** Large-sample hydrology dataset — basin attributes + meteorological
  forcings + observed streamflow for many catchments; the benchmark for rainfall-runoff
  modelling (e.g. LSTM vs conceptual models). Regional variants exist (US, GB, CL, BR, ...)
  (verify which one you need).
- **Install:** plain `pandas` (the files are CSV/text); no special API client.
- **Auth/key:** no API key; download the dataset archive from its host and cite it (verify
  the current host/DOI and licence for the variant you use).
- **Minimal snippet:**
  ```python
  import pandas as pd
  # After downloading and extracting the CAMELS archive locally:
  attrs = pd.read_csv("data/camels/camels_attributes.csv")      # (path/schema varies by variant)
  flow = pd.read_csv("data/camels/usgs_streamflow/<gauge>.txt", sep=r"\s+")  # (verify)
  ```
- **Caching tip:** keep the extracted archive read-only and cache derived per-basin frames;
  **split by basin (spatial group)** so basins never cross train/test (leakage). Record the
  gauge ids and the CAMELS variant/DOI in provenance.
- **Synthetic stand-in:** `common/synthetic_climate.py` series can stand in for streamflow-
  like behavior (seasonality + AR(1) persistence + extremes) when prototyping the modelling
  loop before downloading real CAMELS data.

---

## Wrap-up checklist (do this for every real fetch)

- [ ] Wrote a `.meta.json` provenance sidecar (source, query, time/space extent, tool
      version, fetched_at, license, checksum).
- [ ] Cache is keyed on the full query and reused (no silent re-download/overwrite).
- [ ] Auth/key requirement confirmed against the dataset's current docs (marked `(verify)`
      where the flow may have changed).
- [ ] Noted known biases (GBIF sampling bias, Sentinel-2 cloud cover, OpenAQ station gaps)
      before any modelling.
- [ ] If real data was not actually needed, used the `common/` synthetic generator instead.
