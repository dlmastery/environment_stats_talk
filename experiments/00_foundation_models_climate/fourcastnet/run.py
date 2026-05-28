"""Run FourCastNet (AFNO backbone) as a forward-pass smoke test.

HONEST CAVEAT — we do NOT have NERSC HDF5 ERA5 inputs (5 TB archive).
FCN's official inference path requires year-long HDF5 files at exact
NERSC pscratch paths plus orography and stats. We do something simpler:

  1. Download `backbone.ckpt` (~897 MB) from NERSC public portal.
  2. Build the AFNONet (20-channel, 720x1440) and load the state dict.
  3. Feed a single (1, 20, 720, 1440) synthetic input cube (broadcast
     from common.gridded_temperature_field). This is NOT a real ERA5
     initial condition — the output is gibberish-but-shape-correct.
  4. Print output shape + per-channel summary stats.

This proves the FCN model loads + runs end-to-end on this machine.
A real forecast requires the NERSC HDF5 archive (see README_inference.md
and the climate-data-fetch skill).

The 20 input channels (in canonical FCN order) are:
  u10, v10, t2m, sp, mslp,
  t850, u1000, v1000, z1000, u850, v850, z850,
  u500, v500, z500, t500, z50,
  r500, r850, tcwv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Add the FCN repo to sys.path
FCN_REPO = Path(os.environ.get("FCN_REPO", "C:/Users/evija/climate/fm_tmp/FourCastNet")).resolve()
sys.path.insert(0, str(FCN_REPO))

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.synthetic_climate import gridded_temperature_field  # noqa: E402

from networks.afnonet import AFNONet  # noqa: E402

CKPT_PATH = os.environ.get(
    "FCN_CKPT",
    "C:/Users/evija/climate/fm_tmp/fcn_data/backbone.ckpt",
)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


CHANNEL_NAMES = [
    "u10", "v10", "t2m", "sp", "mslp",
    "t850", "u1000", "v1000", "z1000",
    "u850", "v850", "z850",
    "u500", "v500", "z500", "t500", "z50",
    "r500", "r850", "tcwv",
]


def _bilinear(small: np.ndarray, h: int = 720, w: int = 1440) -> np.ndarray:
    src_lat = np.linspace(-90, 90, small.shape[0])
    src_lon = np.linspace(0, 360, small.shape[1], endpoint=False)
    tgt_lat = np.linspace(90, -90, h)
    tgt_lon = np.linspace(0, 360, w, endpoint=False)
    tmp = np.empty((small.shape[0], w), dtype=np.float32)
    for i in range(small.shape[0]):
        tmp[i] = np.interp(tgt_lon, src_lon, small[i].astype(np.float32))
    out = np.empty((h, w), dtype=np.float32)
    for j in range(w):
        out[:, j] = np.interp(tgt_lat, src_lat, tmp[:, j])
    return out


def synth_input() -> np.ndarray:
    """Build a (1, 20, 720, 1440) standardized-ish synthetic input."""
    g = gridded_temperature_field(n_lat=32, n_lon=64, seed=2).field  # deg C
    t2m_k = _bilinear(g) + 273.15  # K
    t2m_norm = (t2m_k - t2m_k.mean()) / (t2m_k.std() + 1e-6)
    # FCN was trained on standardized data. Without the real
    # global_means / global_stds we can only provide a *roughly*
    # standardized cube. All 20 channels are derived from t2m_norm
    # with small scalings (NOT physically meaningful, just to keep
    # the FFT inputs finite).
    cube = np.stack([t2m_norm.astype(np.float32) * (1.0 + 0.05 * i) for i in range(20)], axis=0)
    return cube[np.newaxis, ...]  # (1, 20, 720, 1440)


def load_model() -> torch.nn.Module:
    params = SimpleNamespace(
        N_in_channels=20,
        N_out_channels=20,
        patch_size=8,
        num_blocks=8,
    )
    model = AFNONet(params, img_size=(720, 1440))
    return model


def main() -> int:
    if not os.path.exists(CKPT_PATH):
        print("FCN backbone.ckpt not found:", CKPT_PATH)
        print("Download (~897 MB) from")
        print("  https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("Building AFNONet (20ch, 720x1440)...")
    model = load_model()
    print(" parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    print("Loading checkpoint:", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    # FCN's official ckpt is a {'model_state': ...} with possible 'module.' prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f" missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(" first missing:", missing[:3])
    if unexpected:
        print(" first unexpected:", unexpected[:3])

    model.eval().to(device)

    print("\nBuilding synthetic (1, 20, 720, 1440) input...")
    x_np = synth_input()
    x = torch.from_numpy(x_np).to(device)
    print(" input shape :", tuple(x.shape), " device:", x.device, " dtype:", x.dtype)

    print("\nForward pass...")
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"=== FourCastNet forward pass: {elapsed:.2f}s ===")
    print(" output shape:", tuple(y.shape))

    y_np = y.detach().cpu().numpy()
    stats = {}
    for i, n in enumerate(CHANNEL_NAMES):
        a = y_np[0, i]
        stats[n] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    summary = {
        "elapsed_s": elapsed,
        "model": "FourCastNet AFNONet backbone (20-ch, 720x1440)",
        "device": str(device),
        "input_shape": list(x_np.shape),
        "output_shape": list(y_np.shape),
        "channel_stats": stats,
        "honest_caveat": (
            "Inputs are SYNTHETIC (broadcast 32x64 temperature field). "
            "Output is the model's 6-hour-step forecast on this synthetic field "
            "in standardized space. NOT a real weather forecast. Demonstrates "
            "the FCN pipeline runs end-to-end with the official checkpoint."
        ),
    }
    out_path = RESULTS / "fourcastnet_pipeline_test.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    np.save(RESULTS / "fcn_forecast.npy", y_np)
    for n, s in stats.items():
        print(f" {n:6s} mean={s['mean']:+.3f}  std={s['std']:+.3f}  min={s['min']:+.3f}  max={s['max']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
