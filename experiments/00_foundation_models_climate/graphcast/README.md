# GraphCast — runnable on real 1° ERA5 example

**Status:** RUNNABLE end-to-end. We ran `GraphCast_small` (1.0°, 13
pressure levels) on the official 1-step ERA5 example file from
`gs://dm_graphcast/dataset/`. Produced a real 6-hour forecast in **60 s
on CPU**. See `results/graphcast_small_one_step.json`.

## Reproduce

    python download_assets.py   # ~210 MB total (params + 1 ERA5 step + stats)
    python run.py

## Variables (validated against `graphcast.graphcast`)

- Surface (5): `2m_temperature, mean_sea_level_pressure,
  10m_v_component_of_wind, 10m_u_component_of_wind, total_precipitation_6hr`
- Atmospheric × 13 levels (6): `temperature, geopotential, u/v/vertical
  wind, specific_humidity` @ `{50, 100, 150, 200, 250, 300, 400, 500,
  600, 700, 850, 925, 1000}` hPa
- Static (2): `geopotential_at_surface, land_sea_mask`
- Forcing (5): `toa_incident_solar_radiation` +
  `year_progress_{sin,cos}` + `day_progress_{sin,cos}`
- **2 input time steps** (t-12h, t-6h), 1+ target step.
