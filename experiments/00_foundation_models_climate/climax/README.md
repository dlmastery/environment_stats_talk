# ClimaX — runnable end-to-end on a 32×64 synthetic field

**Status:** RUNNABLE end-to-end. ClimaX's 5.625° grid is **32×64**,
exactly the shape of our `common.gridded_temperature_field`. We loaded
108 M params from the HF checkpoint with **0 missing / 0 unexpected** and
ran a forward pass at lead_time=72h in **0.91 s on RTX 4090**. See
`results/climax_5.625deg_pipeline.json`.

## Reproduce

    pip install timm einops
    curl -L -o 5.625deg.ckpt https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt
    CLIMAX_CKPT=/abs/path/5.625deg.ckpt CLIMAX_REPO=/abs/path/ClimaX python run.py

## 48 input variables (see `configs/global_forecast_climax.yaml`)

land_sea_mask, orography, lattitude, 2m_temperature,
10m_{u,v}_component_of_wind, plus geopotential / u / v / temperature /
relative_humidity / specific_humidity at `[50, 250, 500, 600, 700, 850,
925]` hPa.

Out-vars (the global-forecast head produces all 48 channels; the
training loss is computed on a subset): `geopotential_500,
temperature_850, 2m_temperature, 10m_{u,v}_component_of_wind`.

## Honest caveats

- The synthetic input is one temperature field broadcast across all 48
  channels — *not* a real ERA5 sample. The output is the model's
  transformation of that synthetic field in **standardized space**
  (mean ≈ 0, std ≈ 1 per channel; ClimaX outputs `(x - μ)/σ` residuals).
- The HF checkpoint uses an older field naming (`channel_*`); we rename
  to current (`var_*`) on load.
- Two API-drift patches in the upstream repo were needed for current
  numpy/timm: `np.float` → `float`, `timm.models.layers.helpers` →
  `timm.layers.helpers`, `Block(drop=)` → `Block(proj_drop=)`.
