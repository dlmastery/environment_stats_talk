"""Run Pangu-Weather (24h ONNX) as a pipeline smoke test.

HONEST CAVEAT — we do NOT have real ERA5 input at 0.25 deg / 721x1440 here
(no CDS API key). The Pangu model strictly expects:
  input_upper   : (5, 13, 721, 1440) float32  -- Z, Q, T, U, V at 13 levels
  input_surface : (4,        721, 1440) float32  -- MSLP, U10, V10, T2M

What we do here:
  - tile our synthetic `common.gridded_temperature_field` (32x64) up to
    721x1440 to populate T2M and T (a globally-smooth, climatologically-
    plausible *but obviously not real-weather* field),
  - fill the other vars with plausible climatological constants
    (MSLP=101325 Pa, winds=0, Z = geopotential at level, Q small),
  - run a single 24-hour forward pass via onnxruntime,
  - print/save output shapes + summary stats.

This proves the Pangu-Weather pipeline runs end-to-end on this machine.
It does NOT produce a meaningful forecast — the input is climatology,
the output is a 1-day Pangu rollout of climatology, useful only for
shape/range sanity checking. To get a real forecast, fetch a real ERA5
initial condition (see `climate-data-fetch` skill or `cdsapi`).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make `import common` work when running from any cwd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.synthetic_climate import gridded_temperature_field  # noqa: E402

import onnxruntime as ort  # noqa: E402

MODEL_PATH = os.environ.get(
    "PANGU_MODEL",
    "C:/Users/evija/climate/fm_tmp/pangu_data/pangu_weather_24.onnx",
)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# 13 pressure levels Pangu expects (top -> bottom of atmosphere)
PANGU_LEVELS_HPA = np.array(
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    dtype=np.float32,
)


def _broadcast_field(small: np.ndarray, lat_n: int = 721, lon_n: int = 1440) -> np.ndarray:
    """Tile a coarse synthetic (32,64) field to (721,1440) via interpolation."""
    import numpy as _np
    src_lat = _np.linspace(-90, 90, small.shape[0])
    src_lon = _np.linspace(0, 360, small.shape[1], endpoint=False)
    tgt_lat = _np.linspace(90, -90, lat_n)        # Pangu lat order top-down
    tgt_lon = _np.linspace(0, 360, lon_n, endpoint=False)
    # bilinear via np.interp twice
    tmp = _np.empty((small.shape[0], lon_n), dtype=_np.float32)
    for i in range(small.shape[0]):
        tmp[i] = _np.interp(tgt_lon, src_lon, small[i].astype(_np.float32))
    out = _np.empty((lat_n, lon_n), dtype=_np.float32)
    for j in range(lon_n):
        out[:, j] = _np.interp(tgt_lat, src_lat, tmp[:, j])
    return out


def synth_inputs():
    """Build Pangu-shape inputs from our synthetic climatology field."""
    g = gridded_temperature_field(n_lat=32, n_lon=64, seed=2)  # deg C
    t_surf_C = _broadcast_field(g.field)            # (721,1440) deg C
    t_surf_K = t_surf_C + 273.15

    # Atmospheric temperature (K) -- standard atmosphere approximation:
    # decrease with altitude using ~-6 K per pressure-level-decade is too
    # rough; use simple linear lapse vs height ln(p0/p)*RT/g approximation.
    # We just use a fixed standard-atmosphere-ish profile and add the surface
    # anomaly at each level.
    std_T_K = np.array(
        [288.15, 281.65, 275.15, 268.65, 262.15, 255.65,
         245.15, 230.15, 220.65, 218.15, 212.65, 217.65, 270.65],
        dtype=np.float32,
    )  # 1000..50 hPa rough US Std Atm
    # T(level, lat, lon) = std_T + (t_surf - 288.15) damped with altitude
    anom = (t_surf_K - 288.15).astype(np.float32)
    damp = np.linspace(1.0, 0.0, 13, dtype=np.float32)
    T = std_T_K[:, None, None] + anom[None, :, :] * damp[:, None, None]

    # Geopotential (m^2/s^2) - hypsometric using std atmosphere
    # Z(p) ~ R*T_bar * ln(p_surf/p) (we just synthesize a plausible field)
    Rd, g0 = 287.05, 9.80665
    p_surf = 101325.0
    z_per_level = np.array([
        Rd * std_T_K[i] * np.log(p_surf / (PANGU_LEVELS_HPA[i] * 100.0))
        for i in range(13)
    ], dtype=np.float32)
    Z = np.broadcast_to(z_per_level[:, None, None], (13, 721, 1440)).astype(np.float32)

    # Specific humidity: small near surface, tiny aloft
    Q_profile = np.array(
        [0.012, 0.010, 0.008, 0.005, 0.003, 0.002,
         0.001, 0.0005, 0.0002, 0.0001, 5e-5, 2e-5, 1e-5],
        dtype=np.float32,
    )
    Q = np.broadcast_to(Q_profile[:, None, None], (13, 721, 1440)).astype(np.float32)

    # Winds = 0 placeholder
    U = np.zeros((13, 721, 1440), dtype=np.float32)
    V = np.zeros((13, 721, 1440), dtype=np.float32)

    input_upper = np.stack([Z, Q, T, U, V], axis=0).astype(np.float32)  # (5,13,721,1440)

    # Surface variables: MSLP, U10, V10, T2M
    MSLP = np.full((721, 1440), 101325.0, dtype=np.float32)
    U10 = np.zeros((721, 1440), dtype=np.float32)
    V10 = np.zeros((721, 1440), dtype=np.float32)
    T2M = t_surf_K.astype(np.float32)
    input_surface = np.stack([MSLP, U10, V10, T2M], axis=0)  # (4,721,1440)
    return input_upper, input_surface


def main() -> int:
    if not os.path.exists(MODEL_PATH):
        print("Pangu ONNX file not found:", MODEL_PATH)
        print("Download from https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view")
        return 2

    print("Loading Pangu-Weather 24h ONNX...")
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.enable_mem_reuse = False
    so.intra_op_num_threads = 4
    providers = ort.get_available_providers()
    print(" available providers:", providers)
    sess = ort.InferenceSession(MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
    print(" loaded.")
    print(" model inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
    print(" model outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

    print("\nBuilding synthetic inputs from common.gridded_temperature_field...")
    input_upper, input_surface = synth_inputs()
    print(" input_upper  :", input_upper.shape, input_upper.dtype)
    print(" input_surface:", input_surface.shape, input_surface.dtype)

    print("\nRunning Pangu-Weather +24h forward pass (CPU; this takes a few minutes)...")
    t0 = time.perf_counter()
    out_upper, out_surface = sess.run(
        None, {"input": input_upper, "input_surface": input_surface}
    )
    elapsed = time.perf_counter() - t0
    print(f"\n=== Pangu-Weather +24h forward pass: {elapsed:.1f}s ===")
    print(" output_upper  :", out_upper.shape, out_upper.dtype)
    print(" output_surface:", out_surface.shape, out_surface.dtype)

    # Summarise -- compare T2M before vs after (per variable name from README)
    # input_surface ordering: MSLP, U10, V10, T2M
    # output_surface follows the same ordering
    surf_names = ["MSLP_Pa", "U10_mps", "V10_mps", "T2M_K"]
    upper_names = ["Z_m2s2", "Q_kgkg", "T_K", "U_mps", "V_mps"]
    surface_stats = {}
    for i, n in enumerate(surf_names):
        a, b = input_surface[i], out_surface[i]
        surface_stats[n] = {
            "input_mean": float(a.mean()),
            "output_mean": float(b.mean()),
            "delta_mean": float((b - a).mean()),
            "delta_abs_max": float(np.abs(b - a).max()),
        }
    upper_stats = {}
    for i, n in enumerate(upper_names):
        a, b = input_upper[i], out_upper[i]
        upper_stats[n] = {
            "input_mean": float(a.mean()),
            "output_mean": float(b.mean()),
            "delta_mean": float((b - a).mean()),
            "delta_abs_max": float(np.abs(b - a).max()),
        }
    summary = {
        "elapsed_s": elapsed,
        "model": "pangu_weather_24.onnx",
        "input_shapes": {
            "input_upper": list(input_upper.shape),
            "input_surface": list(input_surface.shape),
        },
        "surface_stats": surface_stats,
        "upper_stats": upper_stats,
        "honest_caveat": (
            "Inputs are SYNTHETIC climatology built from common.gridded_temperature_field. "
            "This is a pipeline / shape smoke test, not a real Pangu forecast."
        ),
    }
    out_path = RESULTS / "pangu_24h_pipeline_test.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    np.save(RESULTS / "t2m_in.npy", input_surface[3])
    np.save(RESULTS / "t2m_out.npy", out_surface[3])
    print("Wrote t2m_in.npy / t2m_out.npy")
    for n, s in surface_stats.items():
        print(f" {n:8s}  in={s['input_mean']:+.3f}  out={s['output_mean']:+.3f}  d={s['delta_mean']:+.4f}  |d|max={s['delta_abs_max']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
