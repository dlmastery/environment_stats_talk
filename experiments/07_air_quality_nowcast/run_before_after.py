"""Orchestrate BEFORE vs AFTER for PM2.5 air-quality nowcasting (Exp 07).

Same synthetic station, same chronological split, same test targets and the same
exceedance threshold for every method -- then writes committed artifacts to
``results/``:

    results/metrics.json        RMSE / MAE / skill-vs-persistence / spike-F1 for each
                                method + the AFTER-vs-BEFORE gains and config.
    results/timeseries_plot.png observed PM2.5 vs persistence/linear/GBM on the test
                                tail, with the exceedance threshold marked.
    results/before_after_bars.png  grouped bars (RMSE and spike-F1: best-BEFORE vs GBM).
    results/summary.md          human-readable table.

This is a case where the win is *driven by the covariates*: the BEFORE baselines see
only PM2.5 history and lag the ventilation-driven spikes; the AFTER GBM reads the
weather (wind, temperature, boundary-layer height) + calendar features and leads
them -- improving RMSE, skill-vs-persistence, AND the exceedance F1. The runner also
records the **history-only ablation** of the AFTER model to show, honestly, that
without the weather covariates the ML model is only ~ persistence.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/07_air_quality_nowcast/run_before_after.py --quick
    python experiments/07_air_quality_nowcast/run_before_after.py
    python experiments/07_air_quality_nowcast/run_before_after.py --model mlp

``--quick`` runs a tiny, fast config (smoke-test the pipeline in well under a
minute). The default finishes in seconds-to-a-minute on CPU and makes the AFTER win
clear. CPU is plenty here; the GBM/MLP are scikit-learn (no GPU needed).
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
from common.synthetic_airquality import (  # noqa: E402
    synthetic_pm25, exceedance_threshold, REAL_DATA_NOTE,
)
from common.synthetic_climate import time_split  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.baseline import run_before  # noqa: E402
from after.gbm_nowcast import GBMConfig, run_after  # noqa: E402
from aq_metrics import all_metrics, rmse  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"
_EXCEEDANCE_Q = 0.90   # define "spike" as PM2.5 above its 90th percentile (test obs)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp07 BEFORE vs AFTER PM2.5 nowcasting")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: short series, small GBM.")
    p.add_argument("--n-days", type=int, default=None,
                   help="Days of synthetic data (default 240, or 90 with --quick).")
    p.add_argument("--freq", choices=["h", "D"], default="h",
                   help="Sampling frequency: hourly (default) or daily.")
    p.add_argument("--model", choices=["gbm", "mlp"], default="gbm",
                   help="AFTER model family (sklearn GradientBoosting or MLP).")
    p.add_argument("--n-lags", type=int, default=6,
                   help="PM2.5 lags for the BEFORE linear AR model.")
    p.add_argument("--no-arima", action="store_true",
                   help="Skip the ARIMA BEFORE baseline (faster).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _config_for(args):
    quick = args.quick
    n_days = args.n_days if args.n_days is not None else (90 if quick else 240)
    cfg = GBMConfig(
        model=args.model,
        n_estimators=80 if quick else 300,
        max_depth=2 if quick else 3,
        use_weather=True,
        seed=args.seed,
    )
    return {"n_days": n_days, "freq": args.freq, "cfg": cfg, "quick": quick,
            "seed": args.seed, "n_lags": args.n_lags,
            "include_arima": not args.no_arima}


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = synthetic_pm25(n_days=conf["n_days"], seed=conf["seed"], freq=conf["freq"])
    train, val, test = time_split(df, 0.7, 0.15)
    cfg = conf["cfg"]

    # BEFORE -- persistence + linear AR (+ ARIMA), PM2.5 history only.
    t0 = time.time()
    before = run_before(pd.concat([train, val]), test, n_lags=conf["n_lags"],
                        include_arima=conf["include_arima"])
    t_before = time.time() - t0

    # AFTER -- GBM/MLP with weather covariates + calendar features.
    t0 = time.time()
    after = run_after(pd.concat([train, val]), test, cfg)
    t_after = time.time() - t0

    # Honest ablation: same AFTER model with weather covariates REMOVED.
    abl_cfg = GBMConfig(**{**cfg.__dict__, "use_weather": False, "feature_names_": []})
    ablation = run_after(pd.concat([train, val]), test, abl_cfg)

    # Align every method on the common trailing test targets so metrics use one set.
    lengths = [len(before[m]["y_true"]) for m in before] + [len(after["y_true"]),
                                                             len(ablation["y_true"])]
    n_common = min(lengths)
    obs = np.asarray(after["y_true"])[-n_common:]

    # Exceedance threshold from the OBSERVED test tail (shared by all methods).
    threshold = exceedance_threshold(obs, _EXCEEDANCE_Q)

    # Persistence RMSE on the common targets = the skill reference for everyone.
    pers_pred = np.asarray(before["persistence"]["y_pred"])[-n_common:]
    rmse_pers = rmse(obs, pers_pred)

    def _scored(y_pred):
        return all_metrics(obs, np.asarray(y_pred)[-n_common:], rmse_pers, threshold)

    before_scored = {name: _scored(before[name]["y_pred"]) for name in before}
    after_scored = _scored(after["y_pred"])
    ablation_scored = _scored(ablation["y_pred"])

    # Pick the strongest BEFORE method by RMSE as the headline comparison.
    best_before_name = min(before_scored, key=lambda k: before_scored[k]["rmse"])
    best_before = before_scored[best_before_name]

    results = {
        "experiment": "07_air_quality_nowcast",
        "config": {
            "n_days": conf["n_days"], "freq": conf["freq"], "quick": conf["quick"],
            "seed": conf["seed"], "n_lags": conf["n_lags"],
            "after_model": cfg.model, "use_weather": True,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_eval_targets": int(n_common),
            "exceedance_quantile": _EXCEEDANCE_Q,
            "exceedance_threshold": round(float(threshold), 3),
            "after_n_features": after["n_features"],
        },
        "before": before_scored,
        "after": {**after_scored, "model": cfg.model, "n_features": after["n_features"]},
        "ablation_history_only": ablation_scored,
        "best_before_method": best_before_name,
        "gains_after_minus_bestbefore": {
            "rmse": after_scored["rmse"] - best_before["rmse"],          # neg = better
            "mae": after_scored["mae"] - best_before["mae"],             # neg = better
            "skill_vs_persistence": after_scored["skill_vs_persistence"]
                                    - best_before["skill_vs_persistence"],
            "spike_f1": after_scored["spike_f1"] - best_before["spike_f1"],  # pos = better
        },
        "wall_time_sec": {"before": round(t_before, 3), "after": round(t_after, 3)},
        "real_data_note": REAL_DATA_NOTE,
    }

    # Plot payloads use the same trailing window for every line.
    plot_payload = {
        "test_index": test.index[-n_common:],
        "obs": obs,
        "persistence": pers_pred,
        "linear_ar": np.asarray(before["linear_ar"]["y_pred"])[-n_common:],
        "after": np.asarray(after["y_pred"])[-n_common:],
        "threshold": threshold,
        "model_label": cfg.model.upper(),
    }
    _write_timeseries(plot_payload)
    _write_bars(best_before, after_scored, best_before_name, cfg.model)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _write_timeseries(payload: dict) -> str:
    plt = plotting.plt
    fig, ax = plotting.new_fig(10.0, 4.4)
    idx = payload["test_index"]
    # Show the last <=480 points so spikes are legible (hourly: ~20 days).
    m = min(len(idx), 480)
    idx = idx[-m:]
    ax.plot(idx, payload["obs"][-m:], color="#202124", lw=1.5, label="Observed PM2.5")
    ax.plot(idx, payload["persistence"][-m:], color=plotting.BEFORE_COLOR, lw=1.0,
            ls=":", label="Before: persistence")
    ax.plot(idx, payload["linear_ar"][-m:], color="#c0392b", lw=1.0, ls="--",
            label="Before: linear AR")
    ax.plot(idx, payload["after"][-m:], color=plotting.AFTER_COLOR, lw=1.3,
            label=f"After: {payload['model_label']} + weather")
    ax.axhline(payload["threshold"], color=plotting.ACCENT, lw=1.0, ls="-.",
               label=f"Exceedance threshold ({payload['threshold']:.0f})")
    ax.set_title("PM2.5 nowcast -- test tail (observed vs models)")
    ax.set_ylabel("PM2.5 (ug/m^3)")
    ax.set_xlabel("time")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    return plotting.save(fig, RESULTS_DIR / "timeseries_plot.png")


def _write_bars(best_before: dict, after_scored: dict, before_name: str,
                model: str) -> str:
    # Two panels would be ideal; the shared helper does grouped bars per metric.
    return plotting.before_after_bars(
        labels=["RMSE (ug/m^3)", "Spike F1 x100"],
        before_vals=[best_before["rmse"], best_before["spike_f1"] * 100.0],
        after_vals=[after_scored["rmse"], after_scored["spike_f1"] * 100.0],
        ylabel="value",
        title=f"PM2.5 nowcast: best-before ({before_name}) vs {model.upper()}+weather",
        path=RESULTS_DIR / "before_after_bars.png",
        lower_is_better=True,  # RMSE lower better; F1 shown x100 -- see caption in README
    )


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _fmt(m: dict) -> str:
    return (f"{m['rmse']:.3f} | {m['mae']:.3f} | {m['skill_vs_persistence']:+.3f} | "
            f"{m['spike_f1']:.3f}")


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    bb = results["best_before_method"]
    g = results["gains_after_minus_bestbefore"]
    lines = [
        "# Experiment 07 -- PM2.5 air-quality nowcast: BEFORE vs AFTER",
        "",
        f"- Synthetic station: {cfg['n_days']} days @ freq=`{cfg['freq']}`  "
        f"|  split: {cfg['n_train']}/{cfg['n_val']}/{cfg['n_test']} (train/val/test)",
        f"- Eval targets (test steps): {cfg['n_eval_targets']}  "
        f"|  AFTER model: `{results['after']['model']}` "
        f"({results['after']['n_features']} features)  |  quick: {cfg['quick']}",
        f"- Exceedance (spike) threshold = {cfg['exceedance_quantile']:.0%} percentile "
        f"of observed test PM2.5 = {cfg['exceedance_threshold']:.1f} ug/m^3",
        "",
        "## Nowcast skill (one-step-ahead, test period)",
        "",
        "| Method | RMSE (ug/m^3) ↓ | MAE ↓ | Skill vs persistence ↑ | Spike F1 ↑ |",
        "|--------|----------------:|------:|-----------------------:|-----------:|",
    ]
    label = {
        "persistence": "Before: persistence",
        "linear_ar": "Before: linear AR",
        "arima": "Before: ARIMA",
    }
    for name, m in results["before"].items():
        lines.append(f"| {label.get(name, name)} | {_fmt(m)} |")
    af = results["after"]
    lines.append(
        f"| **After: {af['model'].upper()} + weather** | "
        f"**{af['rmse']:.3f}** | **{af['mae']:.3f}** | "
        f"**{af['skill_vs_persistence']:+.3f}** | **{af['spike_f1']:.3f}** |")
    abl = results["ablation_history_only"]
    lines.append(f"| _Ablation: {af['model'].upper()} history-only (no weather)_ | {_fmt(abl)} |")
    lines.append(f"| _AFTER − best-before ({bb})_ | "
                 f"{g['rmse']:+.3f} | {g['mae']:+.3f} | "
                 f"{g['skill_vs_persistence']:+.3f} | {g['spike_f1']:+.3f} |")
    lines += [
        "",
        "Skill vs persistence = `1 - RMSE_model/RMSE_persistence` (1 = perfect, 0 = "
        "ties persistence, <0 = worse). Spike F1 = F1 for detecting threshold "
        "exceedances (episodes). The AFTER model reads the **weather covariates** "
        "(wind ventilation, temperature, boundary-layer height) + calendar features, "
        "so it leads the ventilation-driven spikes the history-only baselines lag. "
        "The history-only **ablation** row shows the win is driven by the covariates: "
        "strip them and the ML model falls back toward the classical baselines.",
        "",
        "## Wall time (this machine)",
        "",
        f"- BEFORE: {results['wall_time_sec']['before']:.2f} s   "
        f"|   AFTER: {results['wall_time_sec']['after']:.2f} s",
        "",
        "Artifacts: `timeseries_plot.png`, `before_after_bars.png`, `metrics.json`.",
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
    bb = results["best_before_method"]
    before_best = results["before"][bb]
    af = results["after"]
    abl = results["ablation_history_only"]
    g = results["gains_after_minus_bestbefore"]
    print(f"[exp07] freq={cfg['freq']} n_days={cfg['n_days']} quick={cfg['quick']} "
          f"model={af['model']} n_eval={cfg['n_eval_targets']}")
    print(f"  BEFORE (best={bb:11s}): RMSE={before_best['rmse']:.3f}  "
          f"skill={before_best['skill_vs_persistence']:+.3f}  "
          f"spikeF1={before_best['spike_f1']:.3f}")
    print(f"  AFTER  ({af['model']}+weather)   : RMSE={af['rmse']:.3f}  "
          f"skill={af['skill_vs_persistence']:+.3f}  spikeF1={af['spike_f1']:.3f}")
    print(f"  ABLATION (history-only)   : RMSE={abl['rmse']:.3f}  "
          f"skill={abl['skill_vs_persistence']:+.3f}  spikeF1={abl['spike_f1']:.3f}")
    print(f"  gain (AFTER-bestBEFORE): RMSE {g['rmse']:+.3f}  spikeF1 {g['spike_f1']:+.3f}")
    print(f"[exp07] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
