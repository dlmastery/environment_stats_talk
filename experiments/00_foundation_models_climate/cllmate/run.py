"""CLLMate is a *benchmark + dataset*, not a single released model.

The CLLMate paper (arXiv:2409.19058, EMNLP-2025 main) introduces
the Weather and Climate Event Forecasting benchmark: 26k news articles
aligned with ERA5 reanalysis raster images plus an event knowledge graph.
The repo (https://github.com/hobolee/CLLMate) ships:

  * `data/dataset_cllmate.json` -- the structured event records (1.8 MB),
  * `script/run_internVL.py` -- a benchmark runner that uses the GENERIC
    `OpenGVLab/InternVL3-2B` MLLM (NOT a CLLMate-specific model),
  * NO trained CLLMate-specific weights are shipped.

So "running CLLMate" here means:
  1) Loading the dataset (7,747 events, dict-of-dicts).
  2) Reporting the basic structure -- event types, time/location coverage,
     fields per record.
  3) (Optional) Running InternVL3-2B on one event's image -- we skip that
     in this script because the *images* are not in this repo subtree (only
     the JSON ships); pulling them needs ERA5 + a separate render script.

The end-to-end "ERA5 raster + news article -> next event" task that the
benchmark scores in the paper requires the full image archive and a fine-
tuned MLLM, which is not provided. Downstream code is therefore documented
in docs/FOUNDATION_MODELS.md as RUNNABLE-WITH-EFFORT (real-data fetch + a
larger compute pipeline).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DATASET_JSON = os.environ.get(
    "CLLMATE_JSON",
    "C:/Users/evija/climate/fm_tmp/cllmate_data/dataset_cllmate.json",
)


def main() -> int:
    if not os.path.exists(DATASET_JSON):
        print("Run download_dataset.py first.")
        return 2
    print("Loading dataset...")
    ds = json.load(open(DATASET_JSON, encoding="utf-8"))
    n = len(ds)
    print(f" loaded {n} records")

    sample = ds["0"]
    print(" first record:", sample)

    # Aggregate stats
    events = Counter()
    locs = Counter()
    has_cause = 0
    has_caused_by = 0
    for k, rec in ds.items():
        events[rec.get("event", "?")] += 1
        locs[rec.get("location", "?")] += 1
        if rec.get("cause"):
            has_cause += 1
        if rec.get("caused by"):
            has_caused_by += 1

    summary = {
        "n_records": n,
        "n_distinct_events": len(events),
        "n_distinct_locations": len(locs),
        "top_events": events.most_common(15),
        "top_locations": locs.most_common(15),
        "records_with_cause": has_cause,
        "records_with_caused_by": has_caused_by,
        "fields_per_record": list(sample.keys()),
        "honest_note": (
            "This is the public CLLMate event dataset only. The paper's "
            "benchmark also needs ERA5 raster renderings and a (generic) "
            "MLLM. No CLLMate-specific model weights were released."
        ),
    }
    out_path = RESULTS / "cllmate_dataset_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nTop events:")
    for e, c in events.most_common(10):
        print(f"  {c:5d}  {e}")
    print("\nTop locations:")
    for l, c in locs.most_common(10):
        print(f"  {c:5d}  {l}")
    print(f"\nrecords with cause links: {has_cause} ({has_cause/n*100:.1f}%)")
    print(f"records with caused-by links: {has_caused_by} ({has_caused_by/n*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
