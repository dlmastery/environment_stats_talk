# CLLMate — public dataset + benchmark, no released model

**Status:** PARTIAL. CLLMate (arXiv:2409.19058, EMNLP-2025 main) releases:

- `data/dataset_cllmate.json` — **7,747 structured climate-event
  records**: event, time, location, coordinate, cause/caused-by KG edges,
  news_id, image_path. 1.8 MB. Public, in-repo.
- `script/run_internVL.py` — a benchmark runner that loads the **generic**
  `OpenGVLab/InternVL3-2B` MLLM. There is no CLLMate-specific model.

The paper's "CLLMate" results are a benchmark over 32 generic MLLMs
(open + closed) on the Weather and Climate Event Forecasting task.

## Reproduce dataset inspection

    python run.py

Output (`results/cllmate_dataset_summary.json`): 7,747 records,
top events `heavy rainfall (519), typhoon (315), strong winds (296),
heavy rain (210), high temperature (181)`. 28.6% have downstream
`cause` edges; 46.6% have upstream `caused by` edges.

## To run the actual benchmark

    pip install transformers accelerate
    # Pull InternVL3-2B (~5 GB) and the ERA5 raster image archive
    # (NOT in this repo subtree -- the paper renders them from raw ERA5).
    python script/run_internVL.py  # from the cloned CLLMate repo
