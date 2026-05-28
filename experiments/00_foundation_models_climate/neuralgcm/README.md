# NeuralGCM — runnable end-to-end on real ARCO-ERA5

**Status:** RUNNABLE. We pulled a 5-day real ERA5 slice from the public
ARCO-ERA5 zarr (anonymous HTTPS), regridded 0.25° → 2.8°, encoded the
initial state, and unrolled 4 days × 24 h. Finished in **944 s
(~16 min) on CPU** (no Windows GPU jaxlib wheel). Output in
`results/neuralgcm_4day.json`:

    temperature        shape=(4,37,128,64)  mean=246.30 K  range=[185.98, 310.24]
    geopotential       shape=(4,37,128,64)  mean=122655.6 m²/s²
    u_component_of_wind                  mean=+5.90 m/s    range=[-67.68, +101.03]
    v_component_of_wind                  mean=+0.02 m/s    range=[-64.41, +67.02]
    specific_humidity                    mean=+0.0017 kg/kg

All physically sensible global ranges → end-to-end pipeline works.

## Reproduce

    pip install neuralgcm
    python run.py

## Notes

- gcsfs/gRPC fails on this Windows box with
  `SSL_ERROR_SSL: CERTIFICATE_VERIFY_FAILED` even with truststore +
  certifi. Workaround: use `fsspec.get_mapper("https://storage.
  googleapis.com/...")` for both checkpoint and ERA5 zarr.
- JAX 0.4.30 on Windows is CPU-only; on Linux + GPU the same script
  runs in a few minutes (paper claim).
