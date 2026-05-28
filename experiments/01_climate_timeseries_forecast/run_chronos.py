"""Run zero-shot Chronos on the same Exp01 split and merge into results.

Adds a `foundation_zero_shot` block per horizon in results/metrics.json (without
disturbing the existing persistence/seasonal/SARIMA/LSTM numbers) and writes a
chronos-vs-others bar chart + a short results/foundation_summary.md.

Usage (from repo root):
    python experiments/01_climate_timeseries_forecast/run_chronos.py
    python experiments/01_climate_timeseries_forecast/run_chronos.py --stride 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import daily_temperature, plotting  # noqa: E402


def _import_chronos_module():
    """Import after/chronos_zero_shot.py without making the experiment a package."""
    import importlib.util
    p = Path(__file__).parent / "after" / "chronos_zero_shot.py"
    spec = importlib.util.spec_from_file_location("chronos_zero_shot", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Zero-shot Chronos on Exp01.")
    ap.add_argument("--n-years", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 7, 14])
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--stride", type=int, default=1,
                    help="Stride over test targets (>=1). Higher = faster eval.")
    ap.add_argument("--prefer", choices=["bolt", "t5"], default="bolt")
    args = ap.parse_args(argv)

    df = daily_temperature(n_years=args.n_years, seed=args.seed)
    series = df["t2m"].to_numpy(dtype=np.float32)
    n = len(series)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    print(f"[run_chronos] n={n} train_end={train_end} val_end={val_end} "
          f"test_n={n - val_end} horizons={args.horizons} stride={args.stride}")

    chronos = _import_chronos_module()
    result = chronos.evaluate(
        series=series, train_end=train_end, val_end=val_end,
        horizons=args.horizons, lookback=args.lookback,
        stride=args.stride, prefer=args.prefer,
    )

    # --- merge into the canonical metrics.json without overwriting it ----------
    results_dir = Path(__file__).parent / "results"
    metrics_path = results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    h_block = metrics.setdefault("horizons", {})
    for h_str, vals in result["horizons"].items():
        h_blob = h_block.setdefault(h_str, {})
        fz = h_blob.get("foundation_zero_shot")
        # If a legacy flat-dict blob is still here (from before run_all_foundation
        # was added), migrate it under the model's short name to avoid clobber.
        if isinstance(fz, dict) and "model" in fz and not any(
            isinstance(v, dict) for v in fz.values()
        ):
            fz = {"chronos_t5_small": dict(fz)}
        if not isinstance(fz, dict):
            fz = {}
        fz["chronos_t5_small"] = {
            "status": "ok",
            "model": result["model"], "mode": result["mode"],
            **vals,
        }
        h_blob["foundation_zero_shot"] = fz
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[run_chronos] merged into {metrics_path}")

    # --- comparison bar chart (RMSE per horizon, all available methods) --------
    methods_order = ["persistence", "seasonal_naive", "sarima", "after", "foundation_zero_shot"]
    labels_map = {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal-naive",
        "sarima": "SARIMA",
        "after": "LSTM (4090)",
        "foundation_zero_shot": f"Chronos (zero-shot, {result['mode']})",
    }
    horizons = sorted(h_block.keys(), key=int)
    rmse_by_method = {m: [] for m in methods_order}
    for h in horizons:
        h_blob = h_block[h]
        for m in methods_order:
            if m in ("persistence", "seasonal_naive", "sarima"):
                rmse_by_method[m].append(h_blob.get("before", {}).get(m, {}).get("rmse"))
            elif m == "after":
                rmse_by_method[m].append(h_blob.get("after", {}).get("rmse"))
            else:
                fz = h_blob.get("foundation_zero_shot", {}) or {}
                # new nested layout: {short_name: {rmse: ...}}; old flat: {rmse: ...}
                blob = fz.get("chronos_t5_small", fz)
                rmse_by_method[m].append(blob.get("rmse"))

    fig, ax = plotting.new_fig(7.5, 4.2)
    x = np.arange(len(horizons))
    width = 0.16
    colors = ["#9aa0a6", "#bdbdbd", "#6a4ca8", "#1a73e8", "#188038"]
    for k, (m, vals) in enumerate(rmse_by_method.items()):
        # plot only positions that have a value
        xs = [x[i] + (k - 2) * width for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if ys:
            ax.bar(xs, ys, width, label=labels_map[m], color=colors[k % len(colors)])
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}d" for h in horizons])
    ax.set_ylabel("Test RMSE (°C)  — lower is better")
    ax.set_title("Exp01: zero-shot foundation vs classical & LSTM")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    plot_path = plotting.save(fig, results_dir / "foundation_comparison.png")
    print(f"[run_chronos] wrote {plot_path}")

    # --- short markdown summary ------------------------------------------------
    lines = [
        f"# Exp01 foundation-model results — {result['model']} ({result['mode']})",
        "",
        "| h (days) | Persistence RMSE | SARIMA RMSE | LSTM RMSE | Chronos (zero-shot) RMSE | Chronos skill vs persistence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in horizons:
        b = h_block[h]
        p = (b.get("before", {}).get("persistence") or {}).get("rmse")
        s = (b.get("before", {}).get("sarima") or {}).get("rmse")
        l = (b.get("after") or {}).get("rmse")
        fz = b.get("foundation_zero_shot") or {}
        blob = fz.get("chronos_t5_small", fz)
        c = blob.get("rmse")
        sk = blob.get("skill_vs_persistence")
        def fmt(x): return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {h} | {fmt(p)} | {fmt(s)} | {fmt(l)} | {fmt(c)} | {fmt(sk)} |")
    lines += [
        "",
        "Zero-shot: no fine-tuning on this series. Honest reading: foundation models "
        "are competitive *without* any task-specific training; whether they beat a "
        "well-fit SARIMA on a strongly-seasonal synthetic series is exactly the "
        "question the table answers.",
    ]
    (results_dir / "foundation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[run_chronos] wrote {results_dir / 'foundation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
