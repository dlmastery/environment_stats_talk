"""Run GraphCast_small on the official 1-step ERA5 example bundled with the repo.

This is the honest demonstration: we DON'T have a CDS API key here, so we
use the *official* small 1.0-deg ERA5 example file that DeepMind ships in
gs://dm_graphcast/dataset/. It is a real ERA5 snapshot, just one timestep,
and matches the GraphCast_small checkpoint's resolution (1 deg, 13 pressure
levels). We run one forward pass and report shapes/stats. No SOTA benchmark
claim — the point is that the pipeline really runs end-to-end on this
machine (jax 0.10, numpy 2.2, RTX 4090 Laptop; JAX is CPU here because
jaxlib-cuda is not on Windows pip).

Inputs needed (downloaded once by `download_assets.py`):
  - graphcast_data/params/GraphCast_small - ERA5 ... .npz  (~144 MB)
  - graphcast_data/stats/{diffs_stddev,mean,stddev}_by_level.nc  (~16 KB)
  - graphcast_data/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc (~66 MB)
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import xarray
import jax
import haiku as hk

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)

DATA_ROOT = Path(os.environ.get(
    "GRAPHCAST_DATA",
    "C:/Users/evija/climate/fm_tmp/graphcast_data",
)).resolve()
PARAMS_NAME = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 "
    "- mesh 2to5 - precipitation input and output.npz"
)
DATASET_NAME = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("jax devices:", jax.devices())

    # --- load checkpoint
    ckpt_path = DATA_ROOT / "params" / PARAMS_NAME
    if not ckpt_path.exists():
        print("MISSING:", ckpt_path)
        print("Run download_assets.py first (~210 MB total).")
        return 2
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Loaded checkpoint:", ckpt_path.name)
    print(" model_config:", model_config)

    # --- load stats
    def _load(name):
        return xarray.load_dataset(str(DATA_ROOT / "stats" / name)).compute()
    diffs_stddev_by_level = _load("diffs_stddev_by_level.nc")
    mean_by_level = _load("mean_by_level.nc")
    stddev_by_level = _load("stddev_by_level.nc")

    # --- load example batch
    ds_path = DATA_ROOT / "dataset" / DATASET_NAME
    if not ds_path.exists():
        print("MISSING:", ds_path)
        return 2
    example_batch = xarray.load_dataset(str(ds_path)).compute()
    # The 1-step ERA5 example stores 'time' as int32 hours (0,6,12). GraphCast
    # data_utils expects timedelta64 (h units). Convert.
    if example_batch["time"].dtype.kind in ("i", "u"):
        example_batch = example_batch.assign_coords(
            time=example_batch["time"].astype("timedelta64[h]"),
        )
    print(" example_batch dims:", dict(example_batch.dims))

    # 1-step file has 3 time slots (-12h, -6h, 0h with inputs+1 target slot).
    train_steps = 1
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{train_steps * 6}h"),
        **dataclasses.asdict(task_config),
    )
    print(" eval_inputs dims:", dict(eval_inputs.dims))
    print(" eval_targets dims:", dict(eval_targets.dims))
    print(" eval_forcings dims:", dict(eval_forcings.dims))

    # --- build forward
    def construct_wrapped():
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct_wrapped()(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return lambda **kw: fn(params=params, state=state, rng=jax.random.PRNGKey(0), **kw)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(with_params(jax.jit(run_forward.apply)))

    # NOTE on calling convention: graphcast.rollout.chunked_prediction expects
    # `rng=` kwarg and a callable that DOESN'T require it (params/state baked
    # in). We use the simpler one-step run_forward_jitted directly here.
    t0 = time.perf_counter()
    predictions = run_forward_jitted(
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n=== GraphCast_small forward pass: {elapsed:.1f}s ===")

    # Summarise output
    summary = {
        "elapsed_s": elapsed,
        "jax_backend": str(jax.default_backend()),
        "checkpoint": PARAMS_NAME,
        "dataset": DATASET_NAME,
        "output_variables": list(predictions.data_vars),
        "output_dims": {v: dict(zip(predictions[v].dims, predictions[v].shape)) for v in predictions.data_vars},
    }
    stats = {}
    for v in predictions.data_vars:
        arr = predictions[v].values
        stats[v] = {
            "shape": list(arr.shape),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }
    summary["stats"] = stats
    for v, s in stats.items():
        print(f" {v:30s} shape={s['shape']!s:30s} mean={s['mean']:+.3f} min={s['min']:+.3f} max={s['max']:+.3f}")

    out = RESULTS / "graphcast_small_one_step.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

    # Save the 2m_temperature forecast field as NPY for further plotting
    t2m_path = RESULTS / "t2m_forecast.npy"
    np.save(t2m_path, predictions["2m_temperature"].values)
    print(f"Wrote {t2m_path}  shape={predictions['2m_temperature'].shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
