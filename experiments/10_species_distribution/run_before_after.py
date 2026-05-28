"""Orchestrate BEFORE vs AFTER species distribution modelling (Experiment 10).

Same synthetic presence + background dataset, same train/test split, same
covariates (temperature, precipitation, elevation) -- then two ways to fit a
suitability surface:

    BEFORE  before/glm_logistic.py   logistic regression / MaxEnt-style GLM on
                                     linear covariates + a couple of squared
                                     terms (classical SDM baseline).
    AFTER   after/gbm_sdm.py         sklearn GradientBoosting on the same
                                     covariates, automatic interactions.

Writes committed artifacts to ``results/``:

    results/metrics.json         AUC + Brier + true-suitability correlation for
                                 GLM and GBM, plus prevalence-calibration.
    results/suitability_map.png  three-panel map: true suitability vs GLM vs GBM.
    results/before_after_bars.png AUC + Brier bars (higher / lower is better).
    results/summary.md           human-readable metrics table.

Run from the REPO ROOT so ``import common`` resolves:

    python experiments/10_species_distribution/run_before_after.py --quick
    python experiments/10_species_distribution/run_before_after.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import plotting  # noqa: E402
from common.synthetic_sdm import synthetic_sdm_dataset  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.glm_logistic import run_before  # noqa: E402
from after.gbm_sdm import run_after  # noqa: E402

RESULTS_DIR = _EXP_DIR / "results"


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Exp10 BEFORE vs AFTER species distribution modelling")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke run: fewer presence + background records.")
    p.add_argument("--n-pres", type=int, default=None,
                   help="Number of presence records (default 800, or 250 with --quick).")
    p.add_argument("--n-bg", type=int, default=None,
                   help="Number of background records (default 4000, or 1500 with --quick).")
    p.add_argument("--train-frac", type=float, default=0.7,
                   help="Fraction of records used for training (default 0.7).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gbm-n-est", type=int, default=300,
                   help="GBM n_estimators (default 300, or 150 with --quick).")
    p.add_argument("--gbm-depth", type=int, default=3,
                   help="GBM max_depth (default 3).")
    p.add_argument("--gbm-lr", type=float, default=0.05,
                   help="GBM learning rate (default 0.05).")
    return p.parse_args(argv)


def _config_for(args):
    quick = args.quick
    n_pres = args.n_pres if args.n_pres is not None else (250 if quick else 800)
    n_bg = args.n_bg if args.n_bg is not None else (1500 if quick else 4000)
    gbm_n_est = args.gbm_n_est if not (quick and args.gbm_n_est == 300) else 150
    return {
        "n_pres": n_pres, "n_bg": n_bg, "quick": quick, "seed": args.seed,
        "train_frac": float(args.train_frac),
        "gbm_n_estimators": gbm_n_est,
        "gbm_max_depth": args.gbm_depth,
        "gbm_learning_rate": args.gbm_lr,
    }


def _split_records(records_df: pd.DataFrame, train_frac: float, seed: int):
    """Shuffle (deterministically) and split into train/test."""
    rng = np.random.default_rng(seed + 101)
    idx = np.arange(len(records_df))
    rng.shuffle(idx)
    cut = int(train_frac * len(records_df))
    train = records_df.iloc[idx[:cut]].reset_index(drop=True)
    test = records_df.iloc[idx[cut:]].reset_index(drop=True)
    return train, test


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config_for(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds = synthetic_sdm_dataset(n_pres=conf["n_pres"], n_bg=conf["n_bg"],
                               seed=conf["seed"])
    train, test = _split_records(ds.records_df, conf["train_frac"], conf["seed"])

    # Both classes must be present on each side of the split for AUC/Brier to be
    # meaningful. With shuffling this holds at the configured sizes, but assert it.
    assert train["y_label"].nunique() == 2 and test["y_label"].nunique() == 2, (
        "train/test split lost one of the classes; raise n_pres or n_bg")

    t0 = time.time()
    before = run_before(train, test, ds.grid_df)
    t_before = time.time() - t0

    t0 = time.time()
    after = run_after(
        train, test, ds.grid_df,
        n_estimators=conf["gbm_n_estimators"],
        max_depth=conf["gbm_max_depth"],
        learning_rate=conf["gbm_learning_rate"],
        random_state=conf["seed"],
    )
    t_after = time.time() - t0

    y_te = test["y_label"].to_numpy(dtype=int)
    suit_grid = ds.grid_df["suitability"].to_numpy(dtype=float)

    def _metrics(pred_te: np.ndarray, pred_grid: np.ndarray) -> dict:
        return {
            "auc": float(roc_auc_score(y_te, pred_te)),
            "brier": float(brier_score_loss(y_te, pred_te)),
            "suitability_corr": float(np.corrcoef(suit_grid, pred_grid)[0, 1]),
            "mean_predicted_prevalence": float(np.mean(pred_te)),
            "prevalence_calibration_gap": float(
                abs(np.mean(pred_te) - np.mean(y_te))),
        }

    glm_metrics = _metrics(before["p_pres_test"], before["p_grid"])
    gbm_metrics = _metrics(after["p_pres_test"], after["p_grid"])

    results = {
        "experiment": "10_species_distribution",
        "config": {
            **conf,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "prevalence_train": float(train["y_label"].mean()),
            "prevalence_test": float(test["y_label"].mean()),
            "grid_shape": list(ds.grid_shape),
            "covariates": list(("temperature", "precipitation", "elevation")),
            "before_features": list(before["feature_names"]),
            "after_features": list(after["feature_names"]),
        },
        "before_glm": {
            "metrics": glm_metrics,
            "coefficients": before["coefficients"],
        },
        "after_gbm": {
            "metrics": gbm_metrics,
            "feature_importances": after["feature_importances"],
            "n_estimators": after["n_estimators"],
            "max_depth": after["max_depth"],
            "learning_rate": after["learning_rate"],
        },
        "margins_after_minus_before": {
            "auc": gbm_metrics["auc"] - glm_metrics["auc"],
            "brier": gbm_metrics["brier"] - glm_metrics["brier"],  # negative is good
            "suitability_corr": (gbm_metrics["suitability_corr"]
                                 - glm_metrics["suitability_corr"]),
        },
        "wall_time_sec": {"before": round(t_before, 4),
                          "after": round(t_after, 4)},
        "notes": (
            "AUC: higher is better (1 = perfect ranking, 0.5 = random). "
            "Brier: lower is better (0 = perfect probability). "
            "suitability_corr: Pearson correlation between predicted P(presence) "
            "and the TRUE noise-free suitability on the evaluation grid."
        ),
    }

    # ---- artifacts -------------------------------------------------------- #
    _write_suitability_map(ds, before["p_grid"], after["p_grid"])
    _write_bars(glm_metrics, gbm_metrics)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _write_suitability_map(ds, p_grid_glm: np.ndarray, p_grid_gbm: np.ndarray) -> str:
    plt = plotting.plt
    ny, nx = ds.grid_shape
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    titles = ("TRUE suitability (noise-free)", "Before: GLM (logistic + sq)",
              "After: GBM")
    panels = (ds.suitability_field, p_grid_glm.reshape(ny, nx),
              p_grid_gbm.reshape(ny, nx))
    vmin = 0.0
    vmax = max(0.6, float(ds.suitability_field.max()))
    for ax, t, panel in zip(axes, titles, panels):
        im = ax.imshow(panel, origin="lower",
                       extent=[ds.grid_x[0], ds.grid_x[-1],
                               ds.grid_y[0], ds.grid_y[-1]],
                       cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(t)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Suitability surface: TRUE vs GLM vs GBM", y=1.02)
    return plotting.save(fig, RESULTS_DIR / "suitability_map.png")


def _write_bars(glm_metrics: dict, gbm_metrics: dict) -> str:
    # Two metric pairs on one figure: AUC (higher better) and Brier (lower better).
    labels = ["AUC", "Brier"]
    before_vals = [glm_metrics["auc"], glm_metrics["brier"]]
    after_vals = [gbm_metrics["auc"], gbm_metrics["brier"]]
    plt = plotting.plt
    import numpy as np  # noqa: F811
    fig, ax = plotting.new_fig(7.0, 4.4)
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2, before_vals, w, label="Before: GLM",
                color=plotting.BEFORE_COLOR)
    b2 = ax.bar(x + w / 2, after_vals, w, label="After: GBM",
                color=plotting.AFTER_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score")
    ax.set_title("BEFORE vs AFTER  (AUC: higher is better;  Brier: lower is better)")
    ax.legend(frameon=False)
    # annotate the bars with values
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9)
    return plotting.save(fig, RESULTS_DIR / "before_after_bars.png")


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    glm = results["before_glm"]["metrics"]
    gbm = results["after_gbm"]["metrics"]
    m = results["margins_after_minus_before"]
    lines = [
        "# Experiment 10 - Species distribution modelling: BEFORE vs AFTER",
        "",
        f"- Records: presences={cfg['n_pres']}, background={cfg['n_bg']}, "
        f"train={cfg['n_train']}, test={cfg['n_test']}",
        f"- Prevalence: train={cfg['prevalence_train']:.3f}, "
        f"test={cfg['prevalence_test']:.3f}",
        f"- Covariates: {', '.join(cfg['covariates'])}",
        f"- Grid: {cfg['grid_shape'][0]} x {cfg['grid_shape'][1]} = "
        f"{cfg['grid_shape'][0]*cfg['grid_shape'][1]} cells",
        f"- Quick mode: {cfg['quick']}, seed: {cfg['seed']}",
        "",
        "## Headline metrics",
        "",
        "| Method | AUC (higher better) | Brier (lower better) | "
        "True-suitability corr | Prevalence calibration gap |",
        "|--------|--------------------:|---------------------:|"
        "-----------------------:|---------------------------:|",
        f"| Before: GLM (logistic + sq) | {glm['auc']:.4f} | {glm['brier']:.4f} | "
        f"{glm['suitability_corr']:.4f} | {glm['prevalence_calibration_gap']:.4f} |",
        f"| **After: GBM** | **{gbm['auc']:.4f}** | **{gbm['brier']:.4f}** | "
        f"**{gbm['suitability_corr']:.4f}** | {gbm['prevalence_calibration_gap']:.4f} |",
        "",
        f"AFTER - BEFORE: AUC {m['auc']:+.4f}, Brier {m['brier']:+.4f} "
        f"(negative = better), suitability corr {m['suitability_corr']:+.4f}.",
        "",
        f"Wall time: before {results['wall_time_sec']['before']:.3f}s, "
        f"after {results['wall_time_sec']['after']:.3f}s.",
        "",
        f"_{results['notes']}_",
        "",
        "Artifacts: `metrics.json`, `suitability_map.png`, `before_after_bars.png`.",
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
    glm = results["before_glm"]["metrics"]
    gbm = results["after_gbm"]["metrics"]
    m = results["margins_after_minus_before"]
    print(f"[exp10] quick={cfg['quick']} n_pres={cfg['n_pres']} n_bg={cfg['n_bg']} "
          f"n_train={cfg['n_train']} n_test={cfg['n_test']} "
          f"prev_test={cfg['prevalence_test']:.3f}")
    print("  method     | AUC    | Brier  | suit_corr")
    print(f"  GLM (before) | {glm['auc']:.4f} | {glm['brier']:.4f} | "
          f"{glm['suitability_corr']:.4f}")
    print(f"  GBM (after)  | {gbm['auc']:.4f} | {gbm['brier']:.4f} | "
          f"{gbm['suitability_corr']:.4f}")
    print(f"[exp10] AFTER - BEFORE  AUC {m['auc']:+.4f}  Brier {m['brier']:+.4f}  "
          f"suit_corr {m['suitability_corr']:+.4f}")
    print(f"[exp10] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
