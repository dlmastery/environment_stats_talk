"""Pull the CLLMate event dataset (1.8 MB) from the public GitHub repo."""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

OUT = os.environ.get(
    "CLLMATE_JSON",
    "C:/Users/evija/climate/fm_tmp/cllmate_data/dataset_cllmate.json",
)
URL = "https://raw.githubusercontent.com/hobolee/CLLMate/master/data/dataset_cllmate.json"


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
        print("[exists]", OUT, os.path.getsize(OUT) / 1e6, "MB")
        return 0
    with urllib.request.urlopen(URL, timeout=60) as r, open(OUT, "wb") as w:
        shutil.copyfileobj(r, w, length=1024 * 1024)
    print(f"[ok] {OUT}  {os.path.getsize(OUT)/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
