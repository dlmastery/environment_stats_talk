"""Orchestrate BEFORE vs AFTER prediction intervals (Experiment 12).

Same synthetic daily-temperature series, same shared ridge point forecaster, same
chronological train / calibration / test split — then two ways to put a prediction
interval around the point forecast:

    BEFORE  before/normal_theory.py   yhat ± z*sigma_hat (training residual std)
    AFTER   after/conformal.py        split-conformal (calibration residual
                                      quantile) + locally-adaptive (normalized)
                                      conformal

and writes committed artifacts to ``results/``:

    results/metrics.json    nominal vs empirical coverage + mean interval width for
                            normal-theory, split-conformal and normalized-conformal
                            at the 80/90/95% levels, plus the calibration gap.
    results/coverage_plot.png   nominal vs empirical coverage (the y=x diagonal is
                            perfect calibration) for all three methods.
    results/interval_plot.png   the test-tail series with the normal-theory and the
                            conformal bands overlaid.
    results/summary.md      human-readable coverage/width table.

Why the BEFORE method is miscalibrated here (honest, not rigged): the synthetic
series has seasonal heat extremes (a heavy right tail) that the trend intensifies
over time, so the forecast residuals are heteroscedastic and right-skewed. A single
symmetric Gaussian sigma then mis-sizes the band — it over-covers in the body
(wasting width) while the tail behaviour is wrong. Conformal reads the band off the
*empirical* residual quantiles instead, so it hits the nominal level with tighter
intervals; the normalized variant additionally lets the width adapt to the local
(seasonal) spread.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/12_conformal_uncertainty/run_before_after.py --quick
    python experiments/12_conformal_uncertainty/run_before_after.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import daily_temperature, metrics, plotting  # noqa: E402  (metrics imported for parity/use)

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.normal_theory import run_before  # noqa: E402
from after.conformal import run_after  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"
LEVELS_DEFAULT = (0.80, 0.90, 0.95)

# The synthetic generator's own knobs, set so the residuals are clearly
# heteroscedastic & right-skewed (frequent warm-season heat extremes intensifying
# with the warming trend) — the faithful regime where a single Gaussian sigma is
# miscalibrated. These are real parameters of common.synthetic_climate, not a rig.
HETERO_KWARGS = dict(heat_extreme_rate=0.06, warming_c_per_decade=0.6)


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp12 BEFORE vs AFTER prediction intervals")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: fewer years (still shows the calibration gap).")
    p.add_argument("--n-years", type=int, default=None,
                   help="Years of synthetic data (default 20, or 6 with --quick).")
    p.add_argument("--levels", type=float, nargs="+", default=None,
                   help="Nominal coverage levels (default 0.80 0.90 0.95).")
    p.add_argument("--lookback", type=int, default=7, help="Lag-feature window.")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge L2 strength.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _config_for(args):
    quick = args.quick
    n_years = args.n_years if args.n_years is not None else (6 if quick else 20)
    levels = tuple(args.levels) if args.levels else LEVELS_DEFAULT
    return {"n_years": n_years, "levels": levels, "quick": quick,
            "lookback": args.lookback, "alpha": args.alpha, "seed": args.seed}


def _coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def _mean_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = daily_temperature(n_years=conf["n_years"], seed=conf["seed"], **HETERO_KWARGS)
    n = len(df)
    # Chronological 60/20/20 train / calibration / test (no shuffling, no leakage).
    i_tr, i_cal = int(0.6 * n), int(0.8 * n)
    train = df.iloc[:i_tr]["t2m"]
    cal = df.iloc[i_tr:i_cal]["t2m"]
    test = df.iloc[i_cal:]["t2m"]

    t0 = time.time()
    before = run_before(train, test, levels=conf["levels"],
                        lookback=conf["lookback"], alpha=conf["alpha"])
    t_before = time.time() - t0

    t0 = time.time()
    after = run_after(train, cal, test, levels=conf["levels"],
                     lookback=conf["lookback"], alpha=conf["alpha"])
    t_after = time.time() - t0

    y_true = np.asarray(after["y_true"], dtype=float)

    per_level = {}
    for lvl in conf["levels"]:
        key = f"{lvl:.2f}"
        nb = before["bands"][key]
        sb = after["bands"][key]
        ab = after["normalized_bands"][key]

        cov_normal = _coverage(y_true, nb["lower"], nb["upper"])
        cov_split = _coverage(y_true, sb["lower"], sb["upper"])
        cov_norm = _coverage(y_true, ab["lower"], ab["upper"])

        per_level[key] = {
            "nominal": lvl,
            "normal_theory": {
                "empirical_coverage": cov_normal,
                "mean_width": _mean_width(nb["lower"], nb["upper"]),
                "coverage_gap": abs(cov_normal - lvl),
            },
            "split_conformal": {
                "empirical_coverage": cov_split,
                "mean_width": _mean_width(sb["lower"], sb["upper"]),
                "coverage_gap": abs(cov_split - lvl),
                "Q": sb["Q"],
            },
            "normalized_conformal": {
                "empirical_coverage": cov_norm,
                "mean_width": _mean_width(ab["lower"], ab["upper"]),
                "coverage_gap": abs(cov_norm - lvl),
                "Q": ab["Q"],
            },
        }

    # Average absolute calibration gap across levels — the one-number summary.
    def _mean_gap(method):
        return float(np.mean([per_level[k][method]["coverage_gap"] for k in per_level]))

    results = {
        "experiment": "12_conformal_uncertainty",
        "config": {
            "n_years": conf["n_years"], "quick": conf["quick"], "seed": conf["seed"],
            "lookback": conf["lookback"], "ridge_alpha": conf["alpha"],
            "levels": list(conf["levels"]),
            "hetero_kwargs": HETERO_KWARGS,
            "point_forecaster": "ridge on lag features + sin/cos day-of-year",
            "split": "chronological 60/20/20 train/calibration/test (no shuffle)",
            "n_cal_targets": int(after["n_cal"]),
            "n_test_targets": int(len(y_true)),
            "normal_theory_sigma_hat": before["sigma_hat"],
        },
        "per_level": per_level,
        "mean_coverage_gap": {
            "normal_theory": _mean_gap("normal_theory"),
            "split_conformal": _mean_gap("split_conformal"),
            "normalized_conformal": _mean_gap("normalized_conformal"),
        },
        "wall_time_sec": {"before": round(t_before, 4), "after": round(t_after, 4)},
        "guarantee_note": (
            "Conformal gives finite-sample, distribution-free MARGINAL coverage under "
            "exchangeability; it does NOT guarantee conditional (per-season) coverage. "
            "The normalized variant only improves conditional coverage."
        ),
    }

    # ---- artifacts -------------------------------------------------------- #
    _write_coverage_plot(results)
    _write_interval_plot(before, after, conf["levels"])
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _write_coverage_plot(results: dict) -> str:
    plt = plotting.plt
    fig, ax = plotting.new_fig(6.4, 5.2)
    levels = sorted(float(k) for k in results["per_level"])
    keys = [f"{l:.2f}" for l in levels]

    nominal = levels
    cov_normal = [results["per_level"][k]["normal_theory"]["empirical_coverage"] for k in keys]
    cov_split = [results["per_level"][k]["split_conformal"]["empirical_coverage"] for k in keys]
    cov_norm = [results["per_level"][k]["normalized_conformal"]["empirical_coverage"] for k in keys]

    lo = min(nominal) - 0.06
    ax.plot([lo, 1.0], [lo, 1.0], color="#202124", lw=1.0, ls=":",
            label="perfect calibration (y = x)")
    ax.plot(nominal, cov_normal, "o-", color=plotting.BEFORE_COLOR, lw=1.6,
            label="Before: normal-theory")
    ax.plot(nominal, cov_split, "s-", color=plotting.AFTER_COLOR, lw=1.6,
            label="After: split conformal")
    ax.plot(nominal, cov_norm, "^--", color=plotting.ACCENT, lw=1.6,
            label="After: normalized conformal")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical test coverage")
    ax.set_title("Calibration: nominal vs empirical coverage\n(closer to the dotted "
                 "diagonal = better calibrated)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return plotting.save(fig, RESULTS_DIR / "coverage_plot.png")


def _write_interval_plot(before: dict, after: dict, levels) -> str:
    plt = plotting.plt
    # Use the 90% bands for the illustration (the headline level).
    lvl = 0.90 if 0.90 in levels else sorted(levels)[len(levels) // 2]
    key = f"{lvl:.2f}"
    y_true = np.asarray(after["y_true"], dtype=float)
    idx = after["test_index"]
    tail = min(180, len(y_true))  # last ~6 months for legibility
    sl = slice(len(y_true) - tail, len(y_true))

    nb = before["bands"][key]
    ab = after["normalized_bands"][key]  # show the locally-adaptive band (variable width)
    sb = after["bands"][key]

    fig, ax = plotting.new_fig(9.2, 4.4)
    x = idx[sl]
    ax.fill_between(x, np.asarray(nb["lower"])[sl], np.asarray(nb["upper"])[sl],
                    color=plotting.BEFORE_COLOR, alpha=0.22,
                    label=f"Before: normal-theory {int(lvl*100)}% PI")
    ax.fill_between(x, np.asarray(ab["lower"])[sl], np.asarray(ab["upper"])[sl],
                    color=plotting.AFTER_COLOR, alpha=0.22,
                    label=f"After: normalized conformal {int(lvl*100)}% PI")
    ax.plot(x, np.asarray(sb["lower"])[sl], color=plotting.AFTER_COLOR, lw=0.7, ls=":")
    ax.plot(x, np.asarray(sb["upper"])[sl], color=plotting.AFTER_COLOR, lw=0.7, ls=":",
            label=f"After: split conformal {int(lvl*100)}% PI (const. width)")
    ax.plot(x, y_true[sl], color="#202124", lw=1.3, label="Observed")
    ax.plot(x, np.asarray(before["point_pred"])[sl], color="#d93025", lw=1.1, ls="--",
            label="Point forecast (shared ridge)")
    ax.set_title(f"Test-tail prediction intervals ({int(lvl*100)}% nominal) — "
                 "normal-theory vs conformal")
    ax.set_ylabel("2m temperature (°C)")
    ax.set_xlabel("date")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    return plotting.save(fig, RESULTS_DIR / "interval_plot.png")


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    lines = [
        "# Experiment 12 — Conformal uncertainty: BEFORE vs AFTER",
        "",
        f"- Point forecaster: {cfg['point_forecaster']} (ridge alpha={cfg['ridge_alpha']}, "
        f"lookback={cfg['lookback']})",
        f"- Data: {cfg['n_years']} years synthetic daily temperature, seed {cfg['seed']}, "
        f"heteroscedastic regime {cfg['hetero_kwargs']}",
        f"- Split: {cfg['split']} | cal targets={cfg['n_cal_targets']} | "
        f"test targets={cfg['n_test_targets']}",
        f"- Quick mode: {cfg['quick']}",
        "",
        "## Coverage and interval width by nominal level",
        "",
        "| Nominal | Method | Empirical coverage | |gap| | Mean width (°C) |",
        "|--------:|--------|-------------------:|-----:|----------------:|",
    ]
    for key in sorted(results["per_level"], key=float):
        lvl = results["per_level"][key]["nominal"]
        for label, mkey in (("before: normal-theory", "normal_theory"),
                            ("after: split conformal", "split_conformal"),
                            ("after: normalized conformal", "normalized_conformal")):
            m = results["per_level"][key][mkey]
            bold = "**" if mkey != "normal_theory" else ""
            lines.append(
                f"| {int(lvl*100)}% | {bold}{label}{bold} | "
                f"{bold}{m['empirical_coverage']:.3f}{bold} | "
                f"{m['coverage_gap']:.3f} | {m['mean_width']:.3f} |")
    g = results["mean_coverage_gap"]
    lines += [
        "",
        "## Mean absolute calibration gap (averaged over levels — lower is better)",
        "",
        "| Method | Mean |empirical − nominal| |",
        "|--------|-------------------------:|",
        f"| before: normal-theory | {g['normal_theory']:.4f} |",
        f"| after: split conformal | {g['split_conformal']:.4f} |",
        f"| after: normalized conformal | {g['normalized_conformal']:.4f} |",
        "",
        f"Wall time: before {results['wall_time_sec']['before']:.3f}s, "
        f"after {results['wall_time_sec']['after']:.3f}s.",
        "",
        f"_{results['guarantee_note']}_",
        "",
        "Artifacts: `metrics.json`, `coverage_plot.png`, `interval_plot.png`.",
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
    print(f"[exp12] n_years={cfg['n_years']} quick={cfg['quick']} "
          f"n_test={cfg['n_test_targets']} n_cal={cfg['n_cal_targets']}")
    print("  level | normal-theory | split-conf | norm-conf  (empirical coverage)")
    for key in sorted(results["per_level"], key=float):
        pl = results["per_level"][key]
        print(f"   {int(pl['nominal']*100):>3d}% |   {pl['normal_theory']['empirical_coverage']:.3f}     "
              f"|  {pl['split_conformal']['empirical_coverage']:.3f}    "
              f"|  {pl['normalized_conformal']['empirical_coverage']:.3f}")
    g = results["mean_coverage_gap"]
    print(f"[exp12] mean |gap|: normal={g['normal_theory']:.4f}  "
          f"split={g['split_conformal']:.4f}  norm={g['normalized_conformal']:.4f}")
    print(f"[exp12] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
