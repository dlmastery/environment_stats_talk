"""Run NeuralGCM 2.8 deg on a real (anonymous) ARCO-ERA5 slice.

NeuralGCM is hybrid ML + physics, and the 2.8-deg variant runs on CPU in
a few minutes. We follow the official `inference_demo.ipynb` exactly:
  1. Load `v1/deterministic_2_8_deg.pkl` from gs://neuralgcm/models/
     (anonymous, 58 MB).
  2. Read a 4-day ERA5 slice from gs://gcp-public-data-arco-era5/...
     (anonymous Zarr, lazy; ~few hundred MB pulled).
  3. Regrid + encode + unroll 4 days.

If anonymous GCS / network is unavailable, this will fail clearly and the
NeuralGCM section in docs/FOUNDATION_MODELS.md will reflect that.

The 2.8 deg run was reported in the paper as fast enough on CPU for a
demonstration; on this Windows / CPU box it takes ~5-10 min.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Point gRPC (used by gcsfs's google-cloud-storage backend) at certifi's CA
# bundle. Without this we get SSL_ERROR_SSL: CERTIFICATE_VERIFY_FAILED.
try:
    import certifi
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", certifi.where())
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import fsspec
import jax
import xarray
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

import neuralgcm

CHECKPOINT_PATH = os.environ.get(
    "NEURALGCM_CKPT",
    "C:/Users/evija/climate/fm_tmp/neuralgcm_data/deterministic_2_8_deg.pkl",
)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("jax devices:", jax.devices())
    print("Loading NeuralGCM checkpoint ...")
    if not os.path.exists(CHECKPOINT_PATH):
        # Pull from public-HTTPS endpoint (gcsfs/gRPC fails due to local SSL).
        import urllib.request as _u
        url = "https://storage.googleapis.com/neuralgcm/models/v1/deterministic_2_8_deg.pkl"
        with _u.urlopen(url, timeout=300) as fr, open(CHECKPOINT_PATH, "wb") as fw:
            while True:
                chunk = fr.read(1024 * 1024)
                if not chunk:
                    break
                fw.write(chunk)
        print(" downloaded.")
    ckpt = pickle.load(open(CHECKPOINT_PATH, "rb"))
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    print(" model loaded.")
    print(" input vars   :", model.input_variables)
    print(" forcing vars :", model.forcing_variables)

    print("\nReading ARCO-ERA5 zarr (anonymous, HTTPS, lazy)...")
    era5_path = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    mapper = fsspec.get_mapper(era5_path)
    full_era5 = xarray.open_zarr(mapper, chunks=None)

    demo_start, demo_end = "2020-02-14", "2020-02-18"
    data_inner_steps = 24
    sliced_era5 = (
        full_era5[model.input_variables + model.forcing_variables]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            time_shift="24 hours",
        )
        .sel(time=slice(demo_start, demo_end, data_inner_steps))
        .compute()
    )
    print(" pulled ERA5 slice:", dict(sliced_era5.sizes))

    # Regrid ERA5 0.25 deg to NeuralGCM 2.8 deg model grid
    print("\nRegridding ERA5 -> model grid...")
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes["latitude"],
        longitude_nodes=full_era5.sizes["longitude"],
        latitude_spacing="equiangular_with_poles",
        longitude_offset=0.0,
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
    eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
    print(" eval_era5 sizes:", dict(eval_era5.sizes))

    inner_steps = 24
    outer_steps = 4 * 24 // inner_steps  # 4 days
    timedelta = np.timedelta64(1, "h") * inner_steps
    times = np.arange(outer_steps) * inner_steps

    print("\nEncoding initial state + unrolling 4 days...")
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
    rng_key = jax.random.key(42)
    initial_state = model.encode(inputs, input_forcings, rng_key)
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

    t0 = time.perf_counter()
    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n=== NeuralGCM 2.8deg 4-day forecast: {elapsed:.1f}s ===")

    predictions_ds = model.data_to_xarray(predictions, times=times)
    print(" predictions vars :", list(predictions_ds.data_vars))
    print(" predictions sizes:", dict(predictions_ds.sizes))

    # Save a t2m-equivalent: NeuralGCM outputs temperature on pressure levels.
    # Snapshot near-surface temperature at level=1000 hPa at last step.
    summary = {
        "elapsed_s": elapsed,
        "model": "neuralgcm v1/deterministic_2_8_deg",
        "input_vars": list(model.input_variables),
        "predictions_vars": list(predictions_ds.data_vars),
        "predictions_sizes": dict(predictions_ds.sizes),
    }

    stats = {}
    for v in predictions_ds.data_vars:
        arr = predictions_ds[v].values
        stats[v] = {
            "shape": list(arr.shape),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }
    summary["stats"] = stats
    (RESULTS / "neuralgcm_4day.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS / 'neuralgcm_4day.json'}")
    for v, s in stats.items():
        print(f" {v:30s} shape={str(s['shape']):28s} mean={s['mean']:+.3f}  min={s['min']:+.3f}  max={s['max']:+.3f}")

    # Save 4-day surface-level temp forecast
    if "temperature" in predictions_ds.data_vars:
        t_arr = predictions_ds["temperature"].values
        np.save(RESULTS / "temperature_forecast.npy", t_arr)
        print(f"Wrote temperature_forecast.npy shape={t_arr.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
