"""Run every available time-series foundation model end-to-end on Exp01.

Mirrors ``run_chronos.py`` but loops over **all** zero-shot foundation models
(Chronos, TimesFM, MOMENT, Moirai, and ClimateLLM if/when public weights ship)
and merges each into ``results/metrics.json`` under
``horizons[h].foundation_zero_shot[<short_name>]``. Gracefully skips any model
whose Python package is not installed.

Usage (from repo root):
    python experiments/01_climate_timeseries_forecast/run_all_foundation.py
    python experiments/01_climate_timeseries_forecast/run_all_foundation.py --stride 2
    python experiments/01_climate_timeseries_forecast/run_all_foundation.py --only timesfm moment

Each model's wall-time and verdict are honest: zero-shot foundations often will
not beat a well-fit SARIMA on a strongly-seasonal synthetic series; we report
what happened.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import daily_temperature, plotting  # noqa: E402

EXP_DIR = Path(__file__).parent
AFTER_DIR = EXP_DIR / "after"
RESULTS_DIR = EXP_DIR / "results"


def _import_after(module_filename: str):
    p = AFTER_DIR / module_filename
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def _try_chronos(series, train_end, val_end, horizons, stride):
    try:
        import chronos  # noqa: F401
    except ImportError as e:
        return None, f"chronos-forecasting not installed: {e}"
    m = _import_after("chronos_zero_shot.py")
    res = m.evaluate(series=series, train_end=train_end, val_end=val_end,
                     horizons=horizons, lookback=96, stride=stride, prefer="bolt")
    return res, None


def _try_timesfm(series, train_end, val_end, horizons, stride):
    try:
        from transformers import TimesFmModelForPrediction  # noqa: F401
    except ImportError as e:
        return None, f"transformers TimesFmModelForPrediction unavailable: {e}"
    m = _import_after("timesfm_zero_shot.py")
    res = m.evaluate(series=series, train_end=train_end, val_end=val_end,
                     horizons=horizons, lookback=512, stride=stride)
    return res, None


def _try_moment(series, train_end, val_end, horizons, stride):
    try:
        import momentfm  # noqa: F401
    except ImportError as e:
        return None, f"momentfm not installed: {e}"
    m = _import_after("moment_zero_shot.py")
    res = m.evaluate(series=series, train_end=train_end, val_end=val_end,
                     horizons=horizons, lookback=512, stride=stride)
    return res, None


def _try_moirai(series, train_end, val_end, horizons, stride):
    try:
        import uni2ts  # noqa: F401
    except ImportError as e:
        return None, f"uni2ts not installed: {e}"
    m = _import_after("moirai_zero_shot.py")
    # Moirai requires context to be a multiple of patch_size (=8 here); use 192.
    res = m.evaluate(series=series, train_end=train_end, val_end=val_end,
                     horizons=horizons, lookback=192, stride=stride)
    return res, None


def _try_climatellm(series, train_end, val_end, horizons, stride):
    """ClimateLLM (arXiv:2502.11059) has no public code/weights release as of
    the talk build date — search of arXiv, OpenReview, and GitHub did not
    surface an official repo. We record this as 'not_available' instead of
    fabricating numbers.
    """
    return None, (
        "ClimateLLM (arXiv:2502.11059) has no public code or pretrained weights "
        "release. Paper only references google-research/weatherbench2 as the "
        "data source; no model GitHub or HF repo is published. Skipping."
    )


# short_name -> (display_label, runner)
RUNNERS: dict[str, tuple[str, Callable]] = {
    "chronos_t5_small": ("Chronos-Bolt-small (zero-shot)", _try_chronos),
    "timesfm": ("TimesFM-2.0-500m (zero-shot)", _try_timesfm),
    "moment": ("MOMENT-1-small (reconstruction zero-shot)", _try_moment),
    "moirai": ("Moirai-1.0-R-small (zero-shot)", _try_moirai),
    "climatellm": ("ClimateLLM (no public release)", _try_climatellm),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zero-shot foundation models on Exp01.")
    ap.add_argument("--n-years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 7, 14])
    ap.add_argument("--stride", type=int, default=2,
                    help="Stride over test targets (>=1).")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=list(RUNNERS.keys()),
                    help="Run a subset (default: all available).")
    args = ap.parse_args(argv)

    df = daily_temperature(n_years=args.n_years, seed=args.seed)
    series = df["t2m"].to_numpy(dtype=np.float32)
    n = len(series)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    print(f"[run_all] n={n} train_end={train_end} val_end={val_end} "
          f"horizons={args.horizons} stride={args.stride}")

    selected = args.only or list(RUNNERS.keys())
    skipped: list[tuple[str, str]] = []

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    h_block = metrics.setdefault("horizons", {})

    # ----- migrate legacy single-foundation blob into nested dict ----------
    for h_str in list(h_block.keys()):
        h_blob = h_block[h_str]
        fz = h_blob.get("foundation_zero_shot")
        if isinstance(fz, dict) and "model" in fz and not any(
            isinstance(v, dict) for v in fz.values()
        ):
            # legacy: {model: ..., rmse: ...} → move under chronos_t5_small
            legacy = dict(fz)
            h_blob["foundation_zero_shot"] = {"chronos_t5_small": legacy}
            print(f"[run_all] migrated legacy foundation_zero_shot @h={h_str}")

    succeeded: dict[str, dict] = {}
    for short in selected:
        label, runner = RUNNERS[short]
        print(f"\n=== {label} ===")
        try:
            result, err = runner(series, train_end, val_end, args.horizons, args.stride)
        except Exception as e:  # pragma: no cover - run-time integration
            result, err = None, f"runtime error: {type(e).__name__}: {e}"
        if result is None:
            print(f"[run_all] SKIP {short}: {err}")
            skipped.append((short, err or "unknown"))
            # Record the skip in metrics so the JSON is self-describing.
            for h in args.horizons:
                h_blob = h_block.setdefault(str(h), {})
                fz = h_blob.setdefault("foundation_zero_shot", {})
                if short not in fz:
                    fz[short] = {"status": "not_available", "reason": err}
            continue
        succeeded[short] = result
        for h_str, vals in result["horizons"].items():
            h_blob = h_block.setdefault(h_str, {})
            fz = h_blob.setdefault("foundation_zero_shot", {})
            fz[short] = {
                "status": "ok",
                "model": result["model"],
                "mode": result.get("mode"),
                "device": result.get("device"),
                **vals,
            }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[run_all] merged into {metrics_path}")

    # --- comparison bar chart: persistence, SARIMA, LSTM + all foundations -
    horizons = sorted(h_block.keys(), key=int)
    classical = ["persistence", "sarima"]
    classical_labels = {"persistence": "Persistence", "sarima": "SARIMA"}
    foundation_keys = sorted(set(
        k for h in horizons
        for k in (h_block[h].get("foundation_zero_shot") or {}).keys()
        if (h_block[h]["foundation_zero_shot"][k] or {}).get("status") == "ok"
    ))
    method_order = classical + ["lstm"] + foundation_keys
    palette = [
        "#9aa0a6", "#6a4ca8", "#1a73e8",  # persistence, sarima, lstm
        "#188038", "#e8710a", "#d93025", "#9334e6", "#0b8043",
    ]

    fig, ax = plotting.new_fig(9.0, 4.6)
    width = 0.8 / max(len(method_order), 1)
    x = np.arange(len(horizons))
    for k, m in enumerate(method_order):
        ys = []
        for h in horizons:
            b = h_block[h]
            if m in classical:
                v = (b.get("before", {}).get(m) or {}).get("rmse")
            elif m == "lstm":
                v = (b.get("after") or {}).get("rmse")
            else:
                v = ((b.get("foundation_zero_shot") or {}).get(m) or {}).get("rmse")
            ys.append(v)
        xs = [x[i] + (k - (len(method_order) - 1) / 2) * width
              for i, v in enumerate(ys) if v is not None]
        ys_plot = [v for v in ys if v is not None]
        if ys_plot:
            label = classical_labels.get(m, "LSTM (4090)" if m == "lstm" else m)
            ax.bar(xs, ys_plot, width, label=label,
                   color=palette[k % len(palette)])
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}d" for h in horizons])
    ax.set_ylabel("Test RMSE (°C) - lower is better")
    ax.set_title("Exp01: zero-shot foundation models vs classical & LSTM")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    plot_path = plotting.save(fig, RESULTS_DIR / "foundation_comparison.png")
    print(f"[run_all] wrote {plot_path}")

    # --- markdown summary table (one row per foundation model) -------------
    lines = [
        "# Exp01 foundation-model zero-shot results",
        "",
        "Every number was produced by `python experiments/01_climate_timeseries_forecast/"
        "run_all_foundation.py` on this machine (Windows, RTX 4090 Laptop). Same "
        "synthetic 20-year daily temperature series and same chronological 70/15/15 "
        "split as the other Exp01 baselines.",
        "",
        "| Model | h=1 RMSE | h=7 RMSE | h=14 RMSE | h=1 skill | h=7 skill | h=14 skill | wall-time (s) | device |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for short in selected:
        label, _ = RUNNERS[short]
        per_h = {}
        device = "—"
        for h in [1, 7, 14]:
            b = ((h_block.get(str(h), {}).get("foundation_zero_shot") or {}).get(short) or {})
            if b.get("status") == "ok":
                per_h[h] = b
                device = b.get("device", "—") or "—"
        if not per_h:
            note = "skipped: package not installed" if short != "climatellm" else "no public release"
            lines.append(f"| {label} | — | — | — | — | — | — | — | {note} |")
            continue
        wall = sum(per_h[h].get("wall_time_sec", 0.0) for h in per_h)

        def fmt(h, key):
            return f"{per_h[h][key]:.3f}" if h in per_h and key in per_h[h] else "—"

        lines.append(
            f"| {label} | {fmt(1, 'rmse')} | {fmt(7, 'rmse')} | {fmt(14, 'rmse')} | "
            f"{fmt(1, 'skill_vs_persistence')} | {fmt(7, 'skill_vs_persistence')} | "
            f"{fmt(14, 'skill_vs_persistence')} | {wall:.2f} | {device} |"
        )
    if skipped:
        lines += ["", "## Skipped"]
        for short, why in skipped:
            lines.append(f"- **{short}**: {why}")
    lines += [
        "",
        "## How to read this",
        "",
        "- *skill_vs_persistence* = 1 - RMSE_model / RMSE_persistence (positive = beats persistence).",
        "- This is **zero-shot**: no fine-tuning on the temperature series.",
        "- On a strongly-seasonal synthetic series, a well-fit SARIMA is hard to beat; "
        "Chronos / TimesFM / MOMENT / Moirai were not exposed to the seasonal "
        "structure at training time. The point is to run the comparison honestly, fast.",
    ]
    (RESULTS_DIR / "foundation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[run_all] wrote {RESULTS_DIR / 'foundation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
