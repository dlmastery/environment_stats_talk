#!/usr/bin/env python
"""Run the whole repo's unit tests, one component per subprocess.

Each experiment is a self-contained mini-project with its own local ``before/``
and ``after/`` packages. Those package names intentionally repeat across
experiments, so collecting them all into a *single* pytest process makes the
names collide in ``sys.modules``. Running each component in its own process
(as a real user would, per-experiment) avoids that entirely and mirrors how the
experiments are meant to be used.

Usage:
    python run_all_tests.py            # run everything
    python run_all_tests.py -k exp01   # pass-through args to each pytest call
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Order: foundations first, then the autoresearch loop, then experiments.
COMPONENTS = [
    "common/tests",
    "autoresearch_env/tests",
    "experiments/01_climate_timeseries_forecast/tests",
    "experiments/02_extreme_value_trends/tests",
    "experiments/03_biodiversity_text_extraction/tests",
    "experiments/04_remote_sensing_landcover/tests",
    "experiments/05_autoresearch_climate/tests",
    "experiments/08_hydrology_streamflow/tests",
    "experiments/12_conformal_uncertainty/tests",
]


def main(argv: list[str]) -> int:
    extra = argv[1:]
    results: list[tuple[str, int]] = []
    for comp in COMPONENTS:
        path = ROOT / comp
        if not path.exists():
            print(f"--- SKIP (missing): {comp}")
            continue
        print(f"\n===== pytest {comp} =====", flush=True)
        rc = subprocess.call(
            [sys.executable, "-m", "pytest", str(path), "-q", *extra],
            cwd=str(ROOT),
        )
        results.append((comp, rc))

    print("\n================ SUMMARY ================")
    failed = 0
    for comp, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
        if rc != 0:
            failed += 1
        print(f"  {status:14s} {comp}")
    print("========================================")
    if failed:
        print(f"{failed} component(s) failed.")
        return 1
    print(f"All {len(results)} components passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
