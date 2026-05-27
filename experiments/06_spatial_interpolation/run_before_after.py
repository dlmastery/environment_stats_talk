"""Orchestrate BEFORE vs AFTER for spatial interpolation (Exp 06).

Same synthetic spatial field, same scattered stations, same dense held-out grid for
every method — then writes committed artifacts to ``results/``:

    results/metrics.json          RMSE/MAE for IDW, kriging, and the covariate-aware
                                  ML model, plus AFTER-vs-BEFORE gains and config.
    results/predicted_surface.png true field vs kriging map vs ML map (+ covariate).
    results/variogram.png         empirical semivariogram of the station values.
    results/before_after_bars.png grouped RMSE/MAE bars (kriging vs ML).
    results/summary.md            human-readable table.

The scientific point: coordinate-only interpolation (IDW, ordinary kriging on (x,y))
models the spatial autocorrelation well, but a large, short-scale part of the field
is driven by a covariate (elevation). Only a model that is GIVEN the covariate can
recover it, so covariate-aware ML attains lower held-out RMSE than kriging — while
kriging retains the calibrated uncertainty surface ML does not provide (an honest
tradeoff, documented in the README and summary).

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/06_spatial_interpolation/run_before_after.py --quick
    python experiments/06_spatial_interpolation/run_before_after.py
    python experiments/06_spatial_interpolation/run_before_after.py --model gbm

``--quick`` runs a tiny, fast config (smoke-test the whole pipeline in a few
seconds). The default config finishes in well under a minute on CPU and makes the
AFTER win clear. Everything is CPU-only and deterministic.
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

from common import plotting  # noqa: E402
from common.metrics import rmse, mae, skill_score  # noqa: E402
from common.synthetic_spatial import (  # noqa: E402
    synthetic_spatial_field, REAL_DATA_NOTE,
)

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.kriging import run_before, empirical_variogram  # noqa: E402
from after.ml_interp import run_after  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp06 BEFORE vs AFTER spatial interpolation")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: fewer stations, coarser grid, fewer trees.")
    p.add_argument("--n-points", type=int, default=None,
                   help="Number of scattered training stations (default 180, or 80 with --quick).")
    p.add_argument("--grid-res", type=int, default=None,
                   help="Dense grid resolution per side (default 45, or 24 with --quick).")
    p.add_argument("--model", choices=["rf", "gbm", "mlp"], default="rf",
                   help="AFTER model backbone (default rf).")
    p.add_argument("--n-estimators", type=int, default=None,
                   help="Trees for the AFTER ensemble (default 400, or 120 with --quick).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _config_for(args):
    quick = args.quick
    n_points = args.n_points if args.n_points is not None else (80 if quick else 180)
    grid_res = args.grid_res if args.grid_res is not None else (24 if quick else 45)
    n_estimators = args.n_estimators if args.n_estimators is not None else (120 if quick else 400)
    return {"n_points": n_points, "grid_res": grid_res, "model": args.model,
            "n_estimators": n_estimators, "quick": quick, "seed": args.seed}


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = synthetic_spatial_field(
        n_points=conf["n_points"], seed=conf["seed"], grid_res=conf["grid_res"],
    )

    # BEFORE — coordinate-only IDW + ordinary kriging (GP).
    t0 = time.time()
    before = run_before(ds, seed=conf["seed"])
    t_before = time.time() - t0

    # AFTER — covariate-aware ML.
    t0 = time.time()
    after = run_after(ds, model=conf["model"],
                      n_estimators=conf["n_estimators"], seed=conf["seed"])
    t_after = time.time() - t0

    krig = before["kriging"]
    idw = before["idw"]

    # univariate informativeness of the covariate (for the summary/README)
    cov = ds.grid_covariate
    val = ds.grid_value
    b1 = np.polyfit(cov, val, 1)
    pred = np.polyval(b1, cov)
    cov_r2 = float(1 - np.sum((val - pred) ** 2) / np.sum((val - val.mean()) ** 2))

    results = {
        "experiment": "06_spatial_interpolation",
        "config": {
            "n_points": conf["n_points"], "grid_res": conf["grid_res"],
            "n_grid": int(ds.grid_coords.shape[0]),
            "model": conf["model"], "n_estimators": conf["n_estimators"],
            "quick": conf["quick"], "seed": conf["seed"], "device": "cpu",
        },
        "covariate_informativeness": {
            "corr_value_covariate": float(np.corrcoef(val, cov)[0, 1]),
            "univariate_r2": cov_r2,
        },
        "before": {
            "idw": {k: v for k, v in idw.items() if k != "y_pred"},
            "kriging": {k: v for k, v in krig.items()
                        if k not in ("y_pred", "y_std")},
        },
        "after": {
            "ml": {k: v for k, v in after.items() if k != "y_pred"},
        },
        "gains_after_vs_kriging": {
            "rmse_abs": after["rmse"] - krig["rmse"],     # negative = AFTER lower error
            "mae_abs": after["mae"] - krig["mae"],
            "rmse_skill": skill_score(after["rmse"], krig["rmse"]),  # 1 - ml/krig
        },
        "uncertainty_note": (
            "Kriging supplies a calibrated predictive-variance surface (mean "
            f"predictive std ~{krig.get('mean_pred_std', float('nan')):.2f}); the "
            "vanilla ML model gives a lower-RMSE point map but no such uncertainty "
            "surface. Lower RMSE is not the whole story."
        ),
        "wall_time_sec": {"before": round(t_before, 3), "after": round(t_after, 3)},
    }

    _write_surface_plot(ds, krig, after)
    _write_variogram_plot(ds)
    _write_bars(krig, after)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _imshow(ax, field2d, ds, title, vmin=None, vmax=None, cmap="viridis"):
    extent = [ds.grid_x[0], ds.grid_x[-1], ds.grid_y[0], ds.grid_y[-1]]
    im = ax.imshow(field2d, origin="lower", extent=extent, aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(False)
    return im


def _write_surface_plot(ds, krig: dict, after: dict) -> str:
    plt = plotting.plt
    ny, nx = ds.grid_shape
    truth = ds.grid_value.reshape(ny, nx)
    krig_map = np.asarray(krig["y_pred"], float).reshape(ny, nx)
    ml_map = np.asarray(after["y_pred"], float).reshape(ny, nx)
    vmin = float(min(truth.min(), krig_map.min(), ml_map.min()))
    vmax = float(max(truth.max(), krig_map.max(), ml_map.max()))

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    im0 = _imshow(axes[0], ds.cov_field, ds, "Covariate field (e.g. elevation)",
                  cmap="terrain")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = _imshow(axes[1], truth, ds, "True field (held-out grid)", vmin, vmax)
    # overlay station locations on the truth map
    axes[1].scatter(ds.train_coords[:, 0], ds.train_coords[:, 1], s=6,
                    c="white", edgecolors="black", linewidths=0.3)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = _imshow(axes[2], krig_map, ds,
                  f"Before: kriging (coords only)\nRMSE={krig['rmse']:.2f}",
                  vmin, vmax)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    im3 = _imshow(axes[3], ml_map, ds,
                  f"After: {after['model'].upper()} + covariate\nRMSE={after['rmse']:.2f}",
                  vmin, vmax)
    fig.colorbar(im3, ax=axes[3], fraction=0.046)
    fig.suptitle("Spatial interpolation — covariate-aware ML recovers terrain detail "
                 "kriging smooths over", fontsize=11)
    return plotting.save(fig, RESULTS_DIR / "predicted_surface.png")


def _write_variogram_plot(ds) -> str:
    centres, gamma, counts = empirical_variogram(
        ds.train_coords, ds.train_value, n_bins=15)
    plt = plotting.plt
    fig, ax = plotting.new_fig(7.0, 4.2)
    good = np.isfinite(gamma)
    ax.plot(centres[good], gamma[good], "o-", color=plotting.ACCENT, lw=1.4,
            label="empirical semivariance γ(h)")
    ax.set_xlabel("lag distance h")
    ax.set_ylabel("semivariance γ(h)")
    ax.set_title("Empirical variogram of station values\n"
                 "(rising γ toward a sill = exploitable spatial autocorrelation)")
    ax.legend(frameon=False)
    return plotting.save(fig, RESULTS_DIR / "variogram.png")


def _write_bars(krig: dict, after: dict) -> str:
    return plotting.before_after_bars(
        labels=["RMSE", "MAE"],
        before_vals=[krig["rmse"], krig["mae"]],
        after_vals=[after["rmse"], after["mae"]],
        ylabel="Error on held-out grid",
        title=f"Spatial interp error: kriging vs {after['model'].upper()}+covariate",
        path=RESULTS_DIR / "before_after_bars.png",
        lower_is_better=True,
    )


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    idw = results["before"]["idw"]
    krig = results["before"]["kriging"]
    ml = results["after"]["ml"]
    g = results["gains_after_vs_kriging"]
    ci = results["covariate_informativeness"]
    top = sorted(ml.get("feature_importance", {}).items(),
                 key=lambda kv: -kv[1])[:3]
    top_str = ", ".join(f"{k} ({v:.2f})" for k, v in top) if top else "n/a"
    lines = [
        "# Experiment 06 — Spatial interpolation: BEFORE vs AFTER",
        "",
        f"- Device: `cpu`  |  stations: {cfg['n_points']}  "
        f"|  held-out grid: {cfg['grid_res']}×{cfg['grid_res']} = {cfg['n_grid']} cells",
        f"- AFTER model: {cfg['model'].upper()} "
        f"(n_estimators={cfg['n_estimators']})  |  quick: {cfg['quick']}  "
        f"|  seed: {cfg['seed']}",
        f"- Covariate informativeness on the grid: "
        f"corr(value, covariate) = {ci['corr_value_covariate']:+.3f}, "
        f"univariate R² = {ci['univariate_r2']:.3f}",
        "",
        "## Interpolation error (held-out dense grid, noise-free truth)",
        "",
        "| Method | Uses covariate? | RMSE ↓ | MAE ↓ |",
        "|--------|:--------------:|------:|------:|",
        f"| Before: IDW (coords only) | no | {idw['rmse']:.3f} | {idw['mae']:.3f} |",
        f"| Before: ordinary kriging / GP (coords only) | no | {krig['rmse']:.3f} | {krig['mae']:.3f} |",
        f"| **After: {cfg['model'].upper()} + covariate** | **yes** | **{ml['rmse']:.3f}** | **{ml['mae']:.3f}** |",
        f"| _AFTER − kriging_ | | {g['rmse_abs']:+.3f} | {g['mae_abs']:+.3f} |",
        "",
        f"AFTER reduces RMSE vs coordinate-only kriging by "
        f"{100*g['rmse_skill']:.1f}% "
        f"(skill score 1 − RMSE_ml/RMSE_krig = {g['rmse_skill']:.3f}). "
        f"Top AFTER features: {top_str}.",
        "",
        "## Honest tradeoff — uncertainty surface",
        "",
        results["uncertainty_note"],
        "",
        "Ordinary kriging is not merely 'the old way': it yields a principled, "
        "spatially-varying **predictive-variance** map (largest far from stations) "
        "that is exactly what many environmental deliverables require (risk maps, "
        "network design, data-assimilation weights). The covariate-aware ML model "
        "here wins on point accuracy but does **not** provide that calibrated "
        "uncertainty for free. The genuinely 'hero' workflow combines them: use the "
        "covariate (regression-kriging / GP with the covariate as an input or mean "
        "function) to get BOTH lower error AND an uncertainty surface.",
        "",
        "## Wall time (this machine)",
        "",
        f"- BEFORE (IDW + kriging fit): {results['wall_time_sec']['before']:.2f} s   "
        f"|   AFTER (ML fit + predict): {results['wall_time_sec']['after']:.2f} s",
        "",
        "Artifacts: `predicted_surface.png`, `variogram.png`, "
        "`before_after_bars.png`, `metrics.json`.",
        "",
        "Swap in real data: " + REAL_DATA_NOTE,
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
    idw = results["before"]["idw"]
    krig = results["before"]["kriging"]
    ml = results["after"]["ml"]
    g = results["gains_after_vs_kriging"]
    print(f"[exp06] model={cfg['model']} n_points={cfg['n_points']} "
          f"grid={cfg['grid_res']}x{cfg['grid_res']} quick={cfg['quick']}")
    print(f"  BEFORE IDW     : RMSE={idw['rmse']:.3f}  MAE={idw['mae']:.3f}")
    print(f"  BEFORE kriging : RMSE={krig['rmse']:.3f}  MAE={krig['mae']:.3f}  "
          f"(coords only; gives an uncertainty surface)")
    print(f"  AFTER  {cfg['model']:4s}    : RMSE={ml['rmse']:.3f}  MAE={ml['mae']:.3f}  "
          f"(+ covariate)")
    print(f"  gain vs kriging: RMSE {g['rmse_abs']:+.3f} "
          f"({100*g['rmse_skill']:.1f}% lower error)")
    print(f"[exp06] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
