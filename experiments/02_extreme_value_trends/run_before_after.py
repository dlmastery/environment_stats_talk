"""Run BEFORE and AFTER for experiment 02 and write all committed artifacts.

Produces, under results/:
  * return_levels.png  — GEV return-level curve with a 95% bootstrap CI band,
                         empirical block-maxima points overlaid.
  * trend_plot.png     — annual maxima time series with the Sen's-slope line.
  * metrics.json       — trend slope, p-value, and 20/50/100-yr return levels
                         with bootstrap CIs (BEFORE empirical + AFTER GEV).
  * summary.md         — a short before/after readout including the validation gate.

Run from the REPO ROOT so `import common` resolves:

    python experiments/02_extreme_value_trends/run_before_after.py            # full
    python experiments/02_extreme_value_trends/run_before_after.py --quick    # fast

`--quick` shrinks the record and bootstrap count for a fast smoke check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
REPO_ROOT = EXP_DIR.parent.parent
# Ensure both the repo root (for `common`) and the experiment dir (for the
# `before`/`after` packages) are importable, whether run from the repo root or
# invoked directly by path.
for _p in (str(REPO_ROOT), str(EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Headless plotting (Agg) — import the shared helper before pyplot anywhere.
from common.plotting import new_fig, save, BEFORE_COLOR, AFTER_COLOR, ACCENT  # noqa: E402

from before import manual_eda  # noqa: E402
from after import agentic_pipeline  # noqa: E402
from after.agentic_pipeline import fit_gev, GEVFit  # noqa: E402
from scipy import stats  # noqa: E402


def _gev_return_curve(c, loc, scale, periods):
    """Return-level curve values for an array of return periods."""
    p = 1.0 - 1.0 / np.asarray(periods, dtype=float)
    return stats.genextreme.ppf(p, c, loc=loc, scale=scale)


def plot_return_levels(after_res: dict, block_maxima: np.ndarray, path: Path) -> str:
    """GEV return-level curve + 95% bootstrap CI band + empirical Gringorten points."""
    g = after_res["gev"]
    periods = np.logspace(np.log10(2), np.log10(200), 60)
    curve = _gev_return_curve(g["c"], g["loc"], g["scale"], periods)

    # CI band: re-bootstrap the curve at a modest count for the shaded envelope.
    bm = np.asarray(block_maxima, dtype=float)
    n_boot = 200 if not after_res["config"]["quick"] else 60
    rng = np.random.default_rng(after_res["config"]["seed"])
    boot_curves = []
    for _ in range(n_boot):
        sample = rng.choice(bm, size=bm.size, replace=True)
        try:
            f = fit_gev(sample)
        except Exception:
            continue
        cv = _gev_return_curve(f.c, f.loc, f.scale, periods)
        if np.all(np.isfinite(cv)):
            boot_curves.append(cv)
    boot_curves = np.asarray(boot_curves)
    lo = np.percentile(boot_curves, 2.5, axis=0)
    hi = np.percentile(boot_curves, 97.5, axis=0)

    # Empirical return periods of the observed block maxima (Gringorten plotting position).
    bm_sorted = np.sort(bm)
    n = bm_sorted.size
    ranks = np.arange(1, n + 1)
    pp = (ranks - 0.44) / (n + 0.12)        # Gringorten non-exceedance prob
    emp_T = 1.0 / (1.0 - pp)

    fig, ax = new_fig(7.2, 4.4)
    ax.fill_between(periods, lo, hi, color=AFTER_COLOR, alpha=0.18,
                    label="95% bootstrap CI")
    ax.plot(periods, curve, color=AFTER_COLOR, lw=2.0, label="GEV return level")
    ax.scatter(emp_T, bm_sorted, s=22, color=ACCENT, zorder=5,
               label="Observed block maxima")
    # Mark the headline 20/50/100-yr points.
    for T, d in after_res["return_levels"].items():
        ax.scatter([T], [d["point"]], s=40, marker="D",
                   edgecolor="k", facecolor="white", zorder=6)
        ax.annotate(f"{T}-yr", (T, d["point"]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Return level — daily precip (mm)")
    ax.set_title("GEV return levels with bootstrap CI (AFTER)")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, path)


def plot_trend(after_res: dict, path: Path) -> str:
    """Annual maxima time series with the Sen's-slope trend line."""
    years = np.asarray(after_res["years"], dtype=float)
    amax = np.asarray(after_res["annual_maxima"], dtype=float)
    mk = after_res["mk_annual_maxima"]
    slope = mk["sen_slope"]
    # Sen intercept: median(y - slope * t_index), t indexed from 0..n-1 (per-year step).
    t_idx = np.arange(len(amax))
    intercept = np.median(amax - slope * t_idx)
    fit_line = intercept + slope * t_idx

    fig, ax = new_fig(7.2, 4.4)
    ax.plot(years, amax, marker="o", ms=4, lw=1.2, color=BEFORE_COLOR,
            label="Annual maxima (Rx1day)")
    ax.plot(years, fit_line, lw=2.2, color=AFTER_COLOR,
            label=f"Sen's slope = {slope:+.3f} mm/yr (p={mk['p_value']:.3f})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual max daily precip (mm)")
    ax.set_title("Annual maxima + Mann-Kendall / Sen's slope")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, path)


def write_metrics(before_res: dict, after_res: dict, path: Path) -> str:
    """Write the consolidated metrics.json."""
    out = {
        "experiment": "02_extreme_value_trends",
        "config": after_res["config"],
        "trend": {
            "annual_maxima": {
                "sen_slope_mm_per_yr": after_res["mk_annual_maxima"]["sen_slope"],
                "mk_z": after_res["mk_annual_maxima"]["z"],
                "p_value": after_res["mk_annual_maxima"]["p_value"],
                "n_blocks": after_res["mk_annual_maxima"]["n"],
            },
            "rx5day": {
                "sen_slope_mm_per_yr": after_res["mk_rx5day"]["sen_slope"],
                "p_value": after_res["mk_rx5day"]["p_value"],
            },
            "r95p": {
                "sen_slope_mm_per_yr": after_res["mk_r95p"]["sen_slope"],
                "p_value": after_res["mk_r95p"]["p_value"],
            },
        },
        "gev": after_res["gev"],
        "return_levels": {
            "after_gev_with_ci": after_res["return_levels"],
            "before_empirical": before_res["empirical_return_levels"],
        },
        "validation": after_res["validation"],
        "trend_table": after_res["trend_table"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return str(path)


def write_summary(before_res: dict, after_res: dict, path: Path,
                  fig_paths: dict) -> str:
    """Write a short before/after markdown summary."""
    mk = after_res["mk_annual_maxima"]
    g = after_res["gev"]
    rl = after_res["return_levels"]
    erl = before_res["empirical_return_levels"]
    v = after_res["validation"]

    lines = []
    lines.append("# Experiment 02 — Extreme-value & trend detection: BEFORE vs AFTER\n")
    lines.append(f"Synthetic daily precipitation, {after_res['config']['n_years']} years, "
                 f"seed {after_res['config']['seed']}, intensification "
                 f"{after_res['config']['intensification_per_decade']}/decade"
                 f"{' (QUICK mode)' if after_res['config']['quick'] else ''}.\n")

    lines.append("## Trend (annual maxima, Rx1day)\n")
    lines.append(f"- Mann-Kendall: z = {mk['z']:+.2f}, p = {mk['p_value']:.4f} "
                 f"over {mk['n']} annual blocks.")
    lines.append(f"- Sen's slope: {mk['sen_slope']:+.4f} mm/yr.\n")

    lines.append("## Return levels (mm)\n")
    lines.append("| T (yr) | BEFORE (empirical) | AFTER (GEV) | AFTER 95% CI |")
    lines.append("|---:|---:|---:|---:|")
    for T in (20, 50, 100):
        d = rl[T]
        lines.append(f"| {T} | {erl[str(T)] if str(T) in erl else erl[T]:.2f} "
                     f"| {d['point']:.2f} | [{d['lo']:.2f}, {d['hi']:.2f}] |")
    lines.append("")

    lines.append("## GEV fit\n")
    lines.append(f"- Method: {g['method']}; mu = {g['loc']:.2f}, sigma = {g['scale']:.2f}, "
                 f"xi = {g['xi']:.3f} (shape c = {g['c']:.3f}), n = {g['n']} blocks.\n")

    lines.append("## Trend table (all indices)\n")
    lines.append("| Index | Sen slope (mm/yr) | MK z | MK p |")
    lines.append("|---|---:|---:|---:|")
    for row in after_res["trend_table"]:
        lines.append(f"| {row['index']} | {row['sen_slope_mm_per_yr']:+.4f} "
                     f"| {row['mk_z']:+.2f} | {row['mk_p']:.4f} |")
    lines.append("")

    lines.append("## Human-in-the-loop validation\n")
    lines.append(f"- Passed (no blocking warnings): **{v['passed']}**")
    for w in v["warnings"]:
        lines.append(f"- [WARN] {w}")
    for note in v["notes"]:
        lines.append(f"- [note] {note}")
    lines.append("")

    lines.append("## Artifacts\n")
    for k, p in fig_paths.items():
        lines.append(f"- `{k}`: `{Path(p).as_posix()}`")
    lines.append("")
    lines.append("## What AFTER adds over BEFORE\n")
    lines.append("- A fitted GEV (MLE) instead of an empirical quantile read-off.")
    lines.append("- Bootstrap confidence intervals on every return level.")
    lines.append("- Significance and a multi-index trend table, not just a point slope.")
    lines.append("- A validation gate (stationarity, autocorrelation, block-count, "
                 "shape sanity, multiple-testing) the analyst must sign off on.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def main():
    ap = argparse.ArgumentParser(description="Run experiment 02 BEFORE+AFTER.")
    ap.add_argument("--quick", action="store_true",
                    help="Fast smoke run (shorter record, fewer bootstrap reps).")
    ap.add_argument("--n-years", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--intensification", type=float, default=0.12,
                    help="Tail intensification per decade.")
    args = ap.parse_args()

    n_years = 25 if args.quick else args.n_years

    print("Running BEFORE (manual EDA) ...")
    before_res = manual_eda.run(
        n_years=n_years, seed=args.seed,
        intensification_per_decade=args.intensification,
    )

    print("Running AFTER (rigorous GEV + trend pipeline) ...")
    after_res = agentic_pipeline.run(
        n_years=n_years, seed=args.seed,
        intensification_per_decade=args.intensification,
        quick=args.quick,
    )

    bm = np.asarray(after_res["annual_maxima"], dtype=float)
    fig_paths = {}
    print("Writing return_levels.png ...")
    fig_paths["return_levels.png"] = plot_return_levels(after_res, bm, RESULTS / "return_levels.png")
    print("Writing trend_plot.png ...")
    fig_paths["trend_plot.png"] = plot_trend(after_res, RESULTS / "trend_plot.png")
    print("Writing metrics.json ...")
    fig_paths["metrics.json"] = write_metrics(before_res, after_res, RESULTS / "metrics.json")
    print("Writing summary.md ...")
    fig_paths["summary.md"] = write_summary(before_res, after_res, RESULTS / "summary.md", fig_paths)

    print("\nDone. Artifacts:")
    for k, p in fig_paths.items():
        print(f"  {k:<20} -> {p}")


if __name__ == "__main__":
    main()
