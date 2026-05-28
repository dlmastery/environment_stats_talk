"""Run ClimaX (5.625 deg, 48-variable global-forecast pretrain) on our synthetic field.

This is the most honest fit in this repo: ClimaX's 5.625-deg variant operates
on a 32x64 lat/lon grid -- the *exact* resolution of our
`common.gridded_temperature_field`. We can therefore run a pure-pipeline test
on synthetic data with zero broadcasting / upsampling.

  - Build ClimaX (32x64, patch 2, embed 1024, depth 8, 48 vars).
  - Load `tungnd/climax/5.625deg.ckpt` (HF, ~432 MB). The HF checkpoint uses
    the OLD field names `channel_embed/channel_query/channel_agg/channel_map`
    while the current arch uses `var_*`; we rename on load.
  - Forge a (1, 48, 32, 64) input from `gridded_temperature_field` (one
    field, replicated across the 48 channels with small per-var scalings).
  - Forward pass with lead_time=72h (the config default) on CUDA.
  - Out-variable subset = ['z500','t850','t2m','u10','v10'].

This is a pipeline run, NOT a real forecast: real ClimaX inputs require all
48 ERA5 fields at the right normalization.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add ClimaX repo
CLIMAX_REPO = Path(os.environ.get("CLIMAX_REPO", "C:/Users/evija/climate/fm_tmp/ClimaX")).resolve()
sys.path.insert(0, str(CLIMAX_REPO / "src"))

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common.synthetic_climate import gridded_temperature_field  # noqa: E402

from climax.arch import ClimaX  # noqa: E402


CKPT_PATH = os.environ.get(
    "CLIMAX_CKPT",
    "C:/Users/evija/climate/fm_tmp/climax_data/5.625deg.ckpt",
)
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# 48 input variables exactly as in the 5.625-deg pretraining config
DEFAULT_VARS = [
    "land_sea_mask", "orography", "lattitude",
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
    "geopotential_50", "geopotential_250", "geopotential_500", "geopotential_600",
    "geopotential_700", "geopotential_850", "geopotential_925",
    "u_component_of_wind_50", "u_component_of_wind_250", "u_component_of_wind_500",
    "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50", "v_component_of_wind_250", "v_component_of_wind_500",
    "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50", "temperature_250", "temperature_500", "temperature_600",
    "temperature_700", "temperature_850", "temperature_925",
    "relative_humidity_50", "relative_humidity_250", "relative_humidity_500",
    "relative_humidity_600", "relative_humidity_700", "relative_humidity_850",
    "relative_humidity_925",
    "specific_humidity_50", "specific_humidity_250", "specific_humidity_500",
    "specific_humidity_600", "specific_humidity_700", "specific_humidity_850",
    "specific_humidity_925",
]
OUT_VARS = ["geopotential_500", "temperature_850", "2m_temperature",
            "10m_u_component_of_wind", "10m_v_component_of_wind"]


def main() -> int:
    if not os.path.exists(CKPT_PATH):
        print("ClimaX 5.625deg.ckpt not found:", CKPT_PATH)
        print("Download (~432 MB) from")
        print("  https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("Building ClimaX (5.625 deg, 32x64, 48 vars, patch=2, embed=1024)...")
    model = ClimaX(
        default_vars=DEFAULT_VARS,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
    )
    print(" parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    print("Loading checkpoint:", CKPT_PATH)
    raw = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = raw.get("state_dict", raw)
    # The HF ckpt stores keys with the "net." prefix (from a Lightning module)
    # AND uses the *original* "channel_*" field names. Map them.
    renamed = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("net."):
            nk = nk[len("net."):]
        nk = (
            nk.replace("channel_embed", "var_embed")
              .replace("channel_query", "var_query")
              .replace("channel_agg", "var_agg")
              .replace("channel_map", "var_map")
        )
        renamed[nk] = v
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    print(f" missing={len(missing)}  unexpected={len(unexpected)}")
    if missing[:3]:
        print(" first missing:", missing[:3])
    if unexpected[:3]:
        print(" first unexpected:", unexpected[:3])

    model.eval().to(device)

    print("\nBuilding synthetic (1, 48, 32, 64) input from gridded_temperature_field...")
    g = gridded_temperature_field(n_lat=32, n_lon=64, seed=2).field  # deg C
    # Normalize roughly (mean 0, std 1) so the model sees in-distribution scale.
    base = (g - g.mean()) / (g.std() + 1e-6)
    # Replicate across 48 channels with small variation
    cube = np.stack(
        [base.astype(np.float32) * (1 + 0.02 * i) for i in range(len(DEFAULT_VARS))],
        axis=0,
    )
    x = torch.from_numpy(cube[None]).to(device)
    print(" input shape:", tuple(x.shape))

    # Other forward() args
    in_vars = DEFAULT_VARS
    out_vars = OUT_VARS
    lead_times = torch.tensor([72.0 / 100.0], device=device)  # normalized lead time

    print("\nForward pass with lead_time=72h...")
    t0 = time.perf_counter()
    with torch.no_grad():
        # signature: forward(x, y, lead_times, variables, out_variables, metric=None, lat=None)
        # In inference mode we don't need y or metrics -- just call the network
        # tokenization+transformer+head explicitly.
        out_transformers = model.forward_encoder(x, lead_times, in_vars)
        preds = model.head(out_transformers)
        # head output is (B, num_patches, V*p*p). Unpatchify.
        preds = model.unpatchify(preds)  # (B, C_out, H, W)
        # Subset to out_vars indices (ClimaX outputs all channels then selects)
        out_idx = [in_vars.index(v) for v in out_vars]
        preds_subset = preds[:, out_idx]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"=== ClimaX forward pass: {elapsed:.3f}s ===")
    print(" preds_full   :", tuple(preds.shape))
    print(" preds_subset :", tuple(preds_subset.shape))

    arr = preds_subset.detach().cpu().numpy()
    stats = {}
    for i, n in enumerate(out_vars):
        a = arr[0, i]
        stats[n] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    summary = {
        "elapsed_s": elapsed,
        "model": "ClimaX 5.625deg pretrain (HF tungnd/climax)",
        "device": str(device),
        "input_shape": list(x.shape),
        "output_shape": list(preds_subset.shape),
        "out_vars": out_vars,
        "stats": stats,
        "honest_caveat": (
            "Inputs are SYNTHETIC: a single 32x64 temperature field broadcast "
            "across all 48 ClimaX channels. Outputs are the model's transform "
            "of this synthetic field in standardized space; NOT a real forecast."
        ),
    }
    (RESULTS / "climax_5.625deg_pipeline.json").write_text(json.dumps(summary, indent=2))
    np.save(RESULTS / "climax_output.npy", arr)
    print(f"\nWrote {RESULTS / 'climax_5.625deg_pipeline.json'}")
    for n, s in stats.items():
        print(f" {n:25s} mean={s['mean']:+.3f}  std={s['std']:+.3f}  min={s['min']:+.3f}  max={s['max']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
