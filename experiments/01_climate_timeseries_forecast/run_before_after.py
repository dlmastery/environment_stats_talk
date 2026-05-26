"""Orchestrate BEFORE vs AFTER for climate time-series forecasting (Experiment 01).

Same synthetic ERA5-like series, same chronological split, same evaluation targets
for every method — then writes committed artifacts to ``results/``:

    results/metrics.json        RMSE / MAE / ACC for each method & horizon, plus
                                the skill score of AFTER vs persistence.
    results/forecast_plot.png   observations vs persistence vs AFTER on the test tail.
    results/before_after_bars.png  grouped bars (persistence vs AFTER RMSE).
    results/summary.md          human-readable table.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/01_climate_timeseries_forecast/run_before_after.py --quick
    python experiments/01_climate_timeseries_forecast/run_before_after.py --epochs 60

``--quick`` runs a tiny, fast configuration (small series, few epochs, SARIMA off)
so you can smoke-test the whole pipeline in well under a minute on CPU. The default
(no flag) is a modest config that finishes in < ~2 min on CPU; the full GPU
headline run is a later pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import daily_temperature, time_split, metrics, plotting  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXP_DIR))
from before.baseline import run_before  # noqa: E402
from after.dl_forecaster import (  # noqa: E402
    TrainConfig, run_after, get_device, zero_shot_foundation_baseline,
)

RESULTS_DIR = _EXP_DIR / "results"
HORIZONS_DEFAULT = (1, 7)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp01 BEFORE vs AFTER climate forecasting")
    p.add_argument("--epochs", type=int, default=None,
                   help="AFTER training epochs (overrides config default).")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: tiny series, few epochs, SARIMA off.")
    p.add_argument("--n-years", type=int, default=None,
                   help="Years of synthetic data (default 20, or 4 with --quick).")
    p.add_argument("--horizons", type=int, nargs="+", default=None,
                   help="Forecast horizons in days (default 1 7).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-foundation", action="store_true",
                   help="Skip the optional zero-shot foundation-model baseline.")
    return p.parse_args(argv)


def _config_for(args):
    """Resolve the run configuration from CLI flags (quick vs modest default)."""
    quick = args.quick
    n_years = args.n_years if args.n_years is not None else (4 if quick else 20)
    horizons = tuple(args.horizons) if args.horizons else (
        (1,) if quick else HORIZONS_DEFAULT
    )
    epochs = args.epochs if args.epochs is not None else (3 if quick else 40)
    include_sarima = not quick
    cfg = TrainConfig(
        lookback=14 if quick else 30,
        hidden=24 if quick else 48,
        epochs=epochs,
        patience=2 if quick else 6,
        seed=args.seed,
    )
    return {
        "n_years": n_years, "horizons": horizons, "include_sarima": include_sarima,
        "cfg": cfg, "quick": quick, "seed": args.seed,
        "use_foundation": (not args.no_foundation) and (not quick),
    }


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = daily_temperature(n_years=conf["n_years"], seed=conf["seed"])
    train, val, test = time_split(df, 0.7, 0.15)
    device = get_device()

    results = {
        "experiment": "01_climate_timeseries_forecast",
        "config": {
            "n_years": conf["n_years"],
            "horizons": list(conf["horizons"]),
            "quick": conf["quick"],
            "seed": conf["seed"],
            "after_cfg": {k: getattr(conf["cfg"], k) for k in
                          ("lookback", "horizon", "hidden", "num_layers",
                           "lr", "batch_size", "epochs", "patience")},
            "device": str(device),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        },
        "horizons": {},
        "wall_time_sec": {},
    }

    plot_payload = None  # for the forecast plot (use h=1 if available, else first h)

    for h in conf["horizons"]:
        cfg = TrainConfig(**{**conf["cfg"].__dict__, "horizon": h})

        t0 = time.time()
        before = run_before(train["t2m"], test["t2m"], h=h,
                            include_sarima=conf["include_sarima"])
        t_before = time.time() - t0

        t0 = time.time()
        after = run_after(train["t2m"], val["t2m"], test["t2m"], cfg)
        t_after = time.time() - t0

        # Align AFTER and BEFORE on a common set of test targets (last N points).
        n_common = min(len(before["persistence"]["y_true"]), len(after["y_true"]))
        pers_pred = np.asarray(before["persistence"]["y_pred"])[-n_common:]
        y_true_c = np.asarray(after["y_true"])[-n_common:]
        after_pred = np.asarray(after["y_pred"])[-n_common:]

        pers_rmse = metrics.rmse(y_true_c, pers_pred)
        after_rmse = metrics.rmse(y_true_c, after_pred)
        skill_vs_persistence = metrics.skill_score(after_rmse, pers_rmse,
                                                    higher_is_better=False)

        # Optional zero-shot foundation baseline (skipped if package absent / quick).
        foundation = None
        if conf["use_foundation"]:
            fz = zero_shot_foundation_baseline(
                pd.concat([train["t2m"], val["t2m"]]), test["t2m"], h=h)
            if fz is not None:
                ft, fp = fz
                foundation = {
                    "rmse": metrics.rmse(ft, fp),
                    "mae": metrics.mae(ft, fp),
                    "acc": metrics.anomaly_correlation(ft, fp),
                    "note": "zero-shot Chronos/TimesFM (no task training)",
                }

        h_block = {
            "before": {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_pred")}
                       for k, v in before.items()},
            "after": {k: v for k, v in after.items() if k not in ("y_true", "y_pred")},
            "skill_score_after_vs_persistence": skill_vs_persistence,
            "n_eval_targets": int(n_common),
        }
        if foundation is not None:
            h_block["foundation_zero_shot"] = foundation
        else:
            h_block["foundation_zero_shot"] = {
                "status": "skipped",
                "reason": "Chronos/TimesFM not installed or disabled - see "
                          "after/dl_forecaster.zero_shot_foundation_baseline.",
            }

        results["horizons"][str(h)] = h_block
        results["wall_time_sec"][str(h)] = {"before": round(t_before, 3),
                                            "after": round(t_after, 3)}

        if plot_payload is None or h == 1:
            plot_payload = {
                "h": h,
                "y_true": y_true_c,
                "persistence": pers_pred,
                "after": after_pred,
                "test_index": test.index[-n_common:],
            }

    # ---- artifacts -------------------------------------------------------- #
    _write_forecast_plot(plot_payload)
    _write_bars(results)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _write_forecast_plot(payload: dict) -> str:
    plt = plotting.plt
    fig, ax = plotting.new_fig(9.0, 4.2)
    tail = min(180, len(payload["y_true"]))  # last ~6 months for legibility
    idx = payload["test_index"][-tail:]
    ax.plot(idx, payload["y_true"][-tail:], color="#202124", lw=1.6, label="Observed")
    ax.plot(idx, payload["persistence"][-tail:], color=plotting.BEFORE_COLOR,
            lw=1.2, ls="--", label="Before: persistence")
    ax.plot(idx, payload["after"][-tail:], color=plotting.AFTER_COLOR,
            lw=1.4, label="After: LSTM")
    ax.set_title(f"Test-tail forecast (h={payload['h']} day)  — observed vs forecasts")
    ax.set_ylabel("2m temperature (°C)")
    ax.set_xlabel("date")
    ax.legend(frameon=False, ncol=3)
    return plotting.save(fig, RESULTS_DIR / "forecast_plot.png")


def _write_bars(results: dict) -> str:
    horizons = list(results["horizons"].keys())
    labels = [f"h={h}d" for h in horizons]
    before_vals, after_vals = [], []
    for h in horizons:
        before_vals.append(results["horizons"][h]["before"]["persistence"]["rmse"])
        after_vals.append(results["horizons"][h]["after"]["rmse"])
    return plotting.before_after_bars(
        labels, before_vals, after_vals,
        ylabel="Test RMSE (°C)",
        title="Climate forecast RMSE: persistence vs LSTM",
        path=RESULTS_DIR / "before_after_bars.png",
        lower_is_better=True,
    )


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    lines = [
        "# Experiment 01 — Climate time-series forecasting: BEFORE vs AFTER",
        "",
        f"- Device: `{cfg['device']}`  |  synthetic years: {cfg['n_years']}  "
        f"|  split: {cfg['n_train']}/{cfg['n_val']}/{cfg['n_test']} (train/val/test)",
        f"- AFTER model: LSTM (hidden={cfg['after_cfg']['hidden']}, "
        f"lookback={cfg['after_cfg']['lookback']}, epochs≤{cfg['after_cfg']['epochs']})",
        f"- Quick mode: {cfg['quick']}",
        "",
        "## Metrics by horizon",
        "",
        "| Horizon | Method | RMSE (°C) | MAE (°C) | ACC |",
        "|--------:|--------|----------:|---------:|----:|",
    ]
    for h, block in results["horizons"].items():
        for name, m in block["before"].items():
            lines.append(f"| {h}d | before: {name} | {m['rmse']:.3f} | "
                         f"{m['mae']:.3f} | {m['acc']:.3f} |")
        a = block["after"]
        lines.append(f"| {h}d | **after: LSTM** | **{a['rmse']:.3f}** | "
                     f"{a['mae']:.3f} | {a['acc']:.3f} |")
        fz = block.get("foundation_zero_shot", {})
        if "rmse" in fz:
            lines.append(f"| {h}d | after: foundation (zero-shot) | {fz['rmse']:.3f} | "
                         f"{fz['mae']:.3f} | {fz['acc']:.3f} |")
        lines.append(f"| {h}d | _skill (LSTM vs persistence)_ | "
                     f"{block['skill_score_after_vs_persistence']:.3f} | | |")
    lines += [
        "",
        "## Wall time (this machine)",
        "",
        "| Horizon | BEFORE (s) | AFTER (s) |",
        "|--------:|-----------:|----------:|",
    ]
    for h, wt in results["wall_time_sec"].items():
        lines.append(f"| {h}d | {wt['before']:.2f} | {wt['after']:.2f} |")
    lines += [
        "",
        "Skill score = 1 − RMSE(after) / RMSE(persistence); >0 means the LSTM beats "
        "the naive persistence reference. Artifacts: `forecast_plot.png`, "
        "`before_after_bars.png`, `metrics.json`.",
        "",
    ]
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    results = run(args)
    # Console digest
    print(f"[exp01] device={results['config']['device']} "
          f"n_years={results['config']['n_years']} quick={results['config']['quick']}")
    for h, block in results["horizons"].items():
        pers = block["before"]["persistence"]["rmse"]
        aft = block["after"]["rmse"]
        sk = block["skill_score_after_vs_persistence"]
        print(f"  h={h}d  persistence RMSE={pers:.3f}  LSTM RMSE={aft:.3f}  "
              f"skill={sk:+.3f}")
    print(f"[exp01] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
