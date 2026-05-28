"""Download GraphCast_small + the matching 1-step ERA5 example file.

Total ~210 MB. Public, anonymous Google Cloud Storage (dm_graphcast bucket).
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.parse
import urllib.request

try:
    import truststore  # Windows SSL truststore fix
    truststore.inject_into_ssl()
except Exception:
    pass

OUTDIR = os.environ.get("GRAPHCAST_DATA", "C:/Users/evija/climate/fm_tmp/graphcast_data")
BASE = "https://storage.googleapis.com/dm_graphcast/"
FILES = [
    "stats/diffs_stddev_by_level.nc",
    "stats/mean_by_level.nc",
    "stats/stddev_by_level.nc",
    "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc",
    "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz",
]


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    for f in FILES:
        out = os.path.join(OUTDIR, f)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"[exists] {f}  ({os.path.getsize(out)/1e6:.1f} MB)")
            continue
        url = BASE + urllib.parse.quote(f)
        print(f"[get  ] {f}")
        with urllib.request.urlopen(url, timeout=180) as r, open(out, "wb") as w:
            shutil.copyfileobj(r, w, length=1024 * 1024)
        print(f"        -> {os.path.getsize(out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
