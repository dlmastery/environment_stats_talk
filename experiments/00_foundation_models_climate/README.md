# 00 — Climate & weather foundation models (audit)

This experiment doesn't fit the standard before/after template — it is
the **audit of every climate / weather foundation model** the talk
transcript names. We checked: can we actually install it, download the
weights, and run a real forward pass on this machine?

See **[../../docs/FOUNDATION_MODELS.md](../../docs/FOUNDATION_MODELS.md)**
for the full per-model write-up — the runnability matrix, exact input
contracts (variables, levels, grid), what we ran, real output stats, and
honest blockers for the not-runnable ones.

Each subdirectory here is one model:

* `graphcast/`   — RUNNABLE end-to-end on real 1° ERA5 (60 s CPU). Real artifact: `results/graphcast_small_one_step.json` + `results/t2m_forecast.npy`.
* `neuralgcm/`   — RUNNABLE-WITH-EFFORT. Pulls real ERA5; JAX-CPU rollout >45 min.
* `pangu_weather/` — RUNNABLE pipeline test (synthetic input, 210 s CPU). Real artifact: `results/pangu_24h_pipeline_test.json` + `results/t2m_{in,out}.npy`.
* `fourcastnet/` — RUNNABLE pipeline test (synthetic input, 3 s on RTX 4090). Real artifact: `results/fourcastnet_pipeline_test.json` + `results/fcn_forecast.npy`.
* `climax/`      — RUNNABLE end-to-end on a 32×64 synthetic field (0.9 s on RTX 4090). Real artifact: `results/climax_5.625deg_pipeline.json` + `results/climax_output.npy`.
* `cllmate/`     — Dataset + benchmark only (no released model). Real artifact: `results/cllmate_dataset_summary.json`.
* `weathernext/` — Service-only via BigQuery / Earth Engine / Vertex AI; no public weights.

Smoke-test imports:

    python -m pytest experiments/00_foundation_models_climate/tests -q

Time to reproduce all runnable pieces end-to-end on this box: ~10 min
(install) + ~5 min (downloads, parallelizable) + ~5 min (running).
