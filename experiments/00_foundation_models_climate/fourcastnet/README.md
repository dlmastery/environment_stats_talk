# FourCastNet — runnable pipeline test on RTX 4090

**Status:** RUNNABLE pipeline test. We downloaded the official 897 MB
`backbone.ckpt` from NERSC, loaded 74.7 M params into `AFNONet(20ch,
720×1440)` with **0 missing / 0 unexpected keys**, and ran a forward
pass in **3.35 s on RTX 4090**. See
`results/fourcastnet_pipeline_test.json` for per-channel stats.

## Reproduce

    pip install timm einops ruamel.yaml
    # Download (897 MB)
    curl -L -o backbone.ckpt https://portal.nersc.gov/project/m4134/FCN_weights_v0/backbone.ckpt
    FCN_CKPT=/abs/path/backbone.ckpt FCN_REPO=/abs/path/FourCastNet python run.py

## Input channels (canonical FCN order)

    u10, v10, t2m, sp, mslp,
    t850, u1000, v1000, z1000,
    u850, v850, z850,
    u500, v500, z500, t500, z50,
    r500, r850, tcwv

All 20 channels on a 720×1440 (0.25°) grid; one step = 6 hours.

## Honest caveat

The synthetic input is a temperature-field-broadcast cube — not a
real ERA5 initial condition. Real forecasts need the NERSC HDF5
archive (~5 TB; see NVlabs/FourCastNet README) plus the matching
`global_means.npy` / `global_stds.npy` / `orography.h5`.
