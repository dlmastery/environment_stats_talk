# Pangu-Weather — runnable pipeline test (synthetic input)

**Status:** RUNNABLE pipeline test on synthetic input (real ERA5 needs
CDS API key, not present here). We downloaded `pangu_weather_24.onnx`
(1.18 GB via Google Drive / `gdown`), built shape-correct inputs from
`common.gridded_temperature_field`, and ran a 24-hour forward pass in
**210 s on CPU** via `onnxruntime`. Output (real, recorded in
`results/pangu_24h_pipeline_test.json`):

    MSLP_Pa   input=101325.000  output=100591.211   Δmean=-733.79   |Δ|max=2497.89
    U10_mps   input=    +0.000  output=    -0.300   Δmean=-0.30    |Δ|max=8.38
    V10_mps   input=    +0.000  output=    +0.196   Δmean=+0.20    |Δ|max=8.49
    T2M_K     input=  +279.004  output=  +278.917   Δmean=-0.09    |Δ|max=15.18

## Reproduce

    pip install onnxruntime gdown truststore
    python -c "import truststore; truststore.inject_into_ssl(); import gdown; gdown.download('https://drive.google.com/uc?id=1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP', 'pangu_weather_24.onnx')"
    PANGU_MODEL=/abs/path/pangu_weather_24.onnx python run.py

## Honest caveat

The output is the 24-hour Pangu forecast of a **synthetic climatology
initial condition**, not real weather. Real forecasts require an actual
ERA5 snapshot at the model's exact 0.25° grid; see the
`climate-data-fetch` skill.
