"""Orchestrate BEFORE vs AFTER for rainfall-runoff streamflow forecasting (Exp 08).

Same synthetic catchment, same chronological split, same test targets for every
method — then writes committed artifacts to ``results/``:

    results/metrics.json        NSE / KGE / RMSE for each method + the AFTER-vs-BEFORE
                                gains and config.
    results/hydrograph.png      observed streamflow vs BOTH models on the test period.
    results/before_after_bars.png  grouped bars (NSE and KGE: linear vs LSTM).
    results/summary.md          human-readable table.

This is the classic case where ML genuinely beats classical hydrology: a linear
regression on lagged precipitation cannot represent the nonlinear, state-dependent
catchment memory (antecedent soil moisture, snow, routing), so it underfits; an
LSTM that carries state captures it. Expect AFTER NSE clearly above BEFORE.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/08_hydrology_streamflow/run_before_after.py --quick
    python experiments/08_hydrology_streamflow/run_before_after.py
    python experiments/08_hydrology_streamflow/run_before_after.py --epochs 120

``--quick`` runs a tiny, fast config (smoke-test the pipeline in well under a
minute on CPU). The default is a modest config that finishes in ~1-3 min and makes
the AFTER win clear; it auto-uses the RTX 4090 when present.
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

from common import plotting  # noqa: E402
from common.synthetic_hydrology import synthetic_catchment, REAL_DATA_NOTE  # noqa: E402
from common.synthetic_climate import time_split  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.linear_baseline import run_before  # noqa: E402
from after.lstm_runoff import LSTMConfig, run_after, get_device  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp08 BEFORE vs AFTER streamflow forecasting")
    p.add_argument("--epochs", type=int, default=None,
                   help="AFTER training epochs (overrides config default).")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: tiny series, few epochs, small LSTM.")
    p.add_argument("--n-years", type=int, default=None,
                   help="Years of synthetic data (default 25, or 5 with --quick).")
    p.add_argument("--n-lags", type=int, default=7,
                   help="Lagged-precip features for the BEFORE linear model.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _config_for(args):
    quick = args.quick
    n_years = args.n_years if args.n_years is not None else (5 if quick else 25)
    epochs = args.epochs if args.epochs is not None else (4 if quick else 60)
    cfg = LSTMConfig(
        lookback=20 if quick else 45,
        hidden=24 if quick else 32,
        epochs=epochs,
        patience=4 if quick else 12,
        lr=1e-3,
        seed=args.seed,
    )
    return {"n_years": n_years, "cfg": cfg, "quick": quick,
            "seed": args.seed, "n_lags": args.n_lags}


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = synthetic_catchment(n_years=conf["n_years"], seed=conf["seed"])
    train, val, test = time_split(df, 0.7, 0.15)
    device = get_device()

    cfg = conf["cfg"]

    # BEFORE — linear regression on lagged precip (+ bucket) fit on train+val.
    t0 = time.time()
    before = run_before(pd.concat([train, val]), test, n_lags=conf["n_lags"])
    t_before = time.time() - t0

    # AFTER — LSTM over lookback window of [precip, temp, pet].
    t0 = time.time()
    after = run_after(train, val, test, cfg)
    t_after = time.time() - t0

    # Align on the common trailing test targets so metrics are computed identically.
    n_common = min(len(before["linear"]["y_true"]), len(after["y_true"]))
    obs = np.asarray(after["y_true"])[-n_common:]
    lin_pred = np.asarray(before["linear"]["y_pred"])[-n_common:]
    lstm_pred = np.asarray(after["y_pred"])[-n_common:]

    from hydro_metrics import all_metrics  # noqa: E402
    lin_m = all_metrics(obs, lin_pred)
    lstm_m = all_metrics(obs, lstm_pred)

    results = {
        "experiment": "08_hydrology_streamflow",
        "config": {
            "n_years": conf["n_years"], "quick": conf["quick"], "seed": conf["seed"],
            "n_lags": conf["n_lags"], "device": str(device),
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_eval_targets": int(n_common),
            "after_cfg": {k: getattr(cfg, k) for k in
                          ("lookback", "hidden", "num_layers", "lr",
                           "batch_size", "epochs", "patience")},
        },
        "before": {
            "linear": lin_m,
            "bucket": {k: v for k, v in before.get("bucket", {}).items()
                       if k not in ("y_true", "y_pred")},
        },
        "after": {
            "lstm": lstm_m,
            "epochs_run": after["epochs_run"],
            "best_val_nse": after["best_val_nse"],
            "device": after["device"],
        },
        "gains_after_minus_before": {
            "nse": lstm_m["nse"] - lin_m["nse"],
            "kge": lstm_m["kge"] - lin_m["kge"],
            "rmse": lstm_m["rmse"] - lin_m["rmse"],  # negative = AFTER lower error
        },
        "wall_time_sec": {"before": round(t_before, 3), "after": round(t_after, 3)},
    }

    plot_payload = {
        "test_index": test.index[-n_common:],
        "obs": obs, "linear": lin_pred, "lstm": lstm_pred,
    }
    _write_hydrograph(plot_payload)
    _write_bars(lin_m, lstm_m)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _write_hydrograph(payload: dict) -> str:
    plt = plotting.plt
    fig, ax = plotting.new_fig(9.5, 4.4)
    idx = payload["test_index"]
    ax.plot(idx, payload["obs"], color="#202124", lw=1.5, label="Observed")
    ax.plot(idx, payload["linear"], color=plotting.BEFORE_COLOR, lw=1.1, ls="--",
            label="Before: linear (lagged precip)")
    ax.plot(idx, payload["lstm"], color=plotting.AFTER_COLOR, lw=1.3,
            label="After: LSTM")
    ax.set_title("Streamflow hydrograph — test period (observed vs models)")
    ax.set_ylabel("Streamflow (mm/day)")
    ax.set_xlabel("date")
    ax.legend(frameon=False, ncol=3)
    return plotting.save(fig, RESULTS_DIR / "hydrograph.png")


def _write_bars(lin_m: dict, lstm_m: dict) -> str:
    return plotting.before_after_bars(
        labels=["NSE", "KGE"],
        before_vals=[lin_m["nse"], lin_m["kge"]],
        after_vals=[lstm_m["nse"], lstm_m["kge"]],
        ylabel="Efficiency (1 = perfect)",
        title="Streamflow skill: linear vs LSTM",
        path=RESULTS_DIR / "before_after_bars.png",
        lower_is_better=False,
    )


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    lin = results["before"]["linear"]
    bucket = results["before"].get("bucket", {})
    lstm = results["after"]["lstm"]
    g = results["gains_after_minus_before"]
    lines = [
        "# Experiment 08 — Rainfall-runoff streamflow forecasting: BEFORE vs AFTER",
        "",
        f"- Device: `{cfg['device']}`  |  synthetic years: {cfg['n_years']}  "
        f"|  split: {cfg['n_train']}/{cfg['n_val']}/{cfg['n_test']} (train/val/test)",
        f"- Eval targets (test days): {cfg['n_eval_targets']}",
        f"- AFTER model: LSTM (hidden={cfg['after_cfg']['hidden']}, "
        f"lookback={cfg['after_cfg']['lookback']}d, epochs≤{cfg['after_cfg']['epochs']}, "
        f"ran {results['after']['epochs_run']})",
        f"- Quick mode: {cfg['quick']}",
        "",
        "## Streamflow skill (test period)",
        "",
        "| Method | NSE ↑ | KGE ↑ | RMSE (mm/day) ↓ |",
        "|--------|------:|------:|----------------:|",
        f"| Before: linear (lagged precip) | {lin['nse']:.3f} | {lin['kge']:.3f} | {lin['rmse']:.3f} |",
    ]
    if bucket:
        lines.append(
            f"| Before: conceptual bucket | {bucket.get('nse', float('nan')):.3f} | "
            f"{bucket.get('kge', float('nan')):.3f} | {bucket.get('rmse', float('nan')):.3f} |")
    lines += [
        f"| **After: LSTM** | **{lstm['nse']:.3f}** | **{lstm['kge']:.3f}** | **{lstm['rmse']:.3f}** |",
        f"| _AFTER − BEFORE (linear)_ | {g['nse']:+.3f} | {g['kge']:+.3f} | {g['rmse']:+.3f} |",
        "",
        "NSE = Nash-Sutcliffe Efficiency, KGE = Kling-Gupta Efficiency (both 1 = "
        "perfect; higher is better). RMSE lower is better. The LSTM carries internal "
        "state across the lookback window, so it captures the nonlinear, "
        "state-dependent catchment memory (antecedent soil moisture, snow, routing) "
        "that the linear model structurally cannot represent.",
        "",
        "## Wall time (this machine)",
        "",
        f"- BEFORE: {results['wall_time_sec']['before']:.2f} s   "
        f"|   AFTER: {results['wall_time_sec']['after']:.2f} s",
        "",
        "Artifacts: `hydrograph.png`, `before_after_bars.png`, `metrics.json`.",
        "",
    ]
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    results = run(args)
    cfg = results["config"]
    lin = results["before"]["linear"]
    lstm = results["after"]["lstm"]
    g = results["gains_after_minus_before"]
    print(f"[exp08] device={cfg['device']} n_years={cfg['n_years']} "
          f"quick={cfg['quick']} epochs_run={results['after']['epochs_run']}")
    print(f"  BEFORE linear : NSE={lin['nse']:.3f}  KGE={lin['kge']:.3f}  "
          f"RMSE={lin['rmse']:.3f}")
    print(f"  AFTER  LSTM   : NSE={lstm['nse']:.3f}  KGE={lstm['kge']:.3f}  "
          f"RMSE={lstm['rmse']:.3f}")
    print(f"  gain (AFTER-BEFORE): NSE {g['nse']:+.3f}  KGE {g['kge']:+.3f}")
    print(f"[exp08] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
