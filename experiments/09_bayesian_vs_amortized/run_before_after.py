"""Orchestrate BEFORE (classical Bayes) vs AFTER (amortized inference) for Exp 09.

Same synthetic hierarchical Normal-Normal dataset, same evaluation (per-station
posterior mean + 95% CI vs the *true* latent ``theta_s``), three methods:

    * BEFORE (closed-form): Normal-Normal conjugate posterior at the KNOWN
      hyperparameters — exact, instant, but assumes you know mu, tau, sigma.
    * BEFORE (MCMC):      from-scratch Metropolis-Hastings over the full joint
      ``(theta_1..S, mu, tau, sigma)`` — what you actually do without PyMC.
    * AFTER (amortized):  small MLP trained once on simulated datasets; scoring
      a fresh dataset is one CPU forward pass.

Honest comparison:
    * MCMC wall time = full per-dataset cost (every new dataset).
    * Amortized wall time = ONE-TIME training + (tiny) per-dataset scoring.
We report both training and scoring times so the trade-off is visible.

Run from REPO ROOT:
    python experiments/09_bayesian_vs_amortized/run_before_after.py --quick
    python experiments/09_bayesian_vs_amortized/run_before_after.py
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
from common.synthetic_bayesian import HierarchyParams, synthetic_station_offsets  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from before.classical_bayes import (  # noqa: E402
    station_sufficient_stats,
    conjugate_posterior_known_hyper,
    mh_sampler,
    summarize_mh,
)
from after.amortized import (  # noqa: E402
    AmortizedConfig, train_amortized, score_amortized, get_device,
)

RESULTS_DIR = _EXP_DIR / "results"


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp09 BEFORE vs AFTER hierarchical Bayes")
    p.add_argument("--quick", action="store_true",
                   help="Tiny config: 12 stations, fewer MH iters, fewer training sims.")
    p.add_argument("--n-stations", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mh-iter", type=int, default=None)
    p.add_argument("--mh-burn", type=int, default=None)
    p.add_argument("--train-sims", type=int, default=None,
                   help="Number of simulated datasets for amortized training.")
    return p.parse_args(argv)


def _config(args):
    quick = args.quick
    S = args.n_stations if args.n_stations is not None else (12 if quick else 30)
    mh_iter = args.mh_iter if args.mh_iter is not None else (1500 if quick else 4000)
    mh_burn = args.mh_burn if args.mh_burn is not None else (500 if quick else 1500)
    train_sims = args.train_sims if args.train_sims is not None else (600 if quick else 4000)
    return {"S": S, "quick": quick, "seed": args.seed,
            "mh_iter": mh_iter, "mh_burn": mh_burn, "train_sims": train_sims}


# --------------------------------------------------------------------------- #
def _coverage(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((truth >= lo) & (truth <= hi)))


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


# --------------------------------------------------------------------------- #
def run(args) -> dict:
    conf = _config(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- evaluation dataset (the one we report coverage on) ----------------- #
    params = HierarchyParams(mu=0.5, tau=1.2, sigma=0.7,
                             n_per_station_min=8, n_per_station_max=20)
    df, truth = synthetic_station_offsets(n_stations=conf["S"], seed=conf["seed"],
                                          params=params)
    suff = station_sufficient_stats(df)
    theta_true = truth["theta"]

    # ----- BEFORE: closed-form (known hyper) --------------------------------- #
    t0 = time.time()
    post_conj = conjugate_posterior_known_hyper(suff, mu=params.mu, tau=params.tau,
                                                sigma=params.sigma)
    t_conj = time.time() - t0

    # ----- BEFORE: MH on the full posterior --------------------------------- #
    t0 = time.time()
    mh = mh_sampler(suff, n_iter=conf["mh_iter"], n_burn=conf["mh_burn"], seed=conf["seed"])
    post_mh = summarize_mh(mh, station_index=suff.index)
    t_mh = time.time() - t0

    # ----- AFTER: amortized -------------------------------------------------- #
    cfg = AmortizedConfig(
        hidden=64 if not conf["quick"] else 32,
        n_layers=3 if not conf["quick"] else 2,
        n_train_datasets=conf["train_sims"],
        batch_datasets=64,
        n_epochs=1,
        seed=conf["seed"],
    )
    device = get_device()
    t0 = time.time()
    model, stats = train_amortized(cfg, device=device)
    t_train = time.time() - t0

    t0 = time.time()
    post_amort = score_amortized(model, suff, stats, device=device)
    t_score = time.time() - t0

    # ----- Metrics ----------------------------------------------------------- #
    def _eval(name, post):
        lo = post["lo95"].to_numpy(); hi = post["hi95"].to_numpy()
        pm = post["post_mean"].to_numpy()
        return {
            "coverage_95": _coverage(lo, hi, theta_true),
            "rmse_to_truth": _rmse(pm, theta_true),
            "mean_interval_width": _interval_width(lo, hi),
        }

    metrics_conj = _eval("conjugate", post_conj)
    metrics_mh = _eval("mh", post_mh)
    metrics_amort = _eval("amortized", post_amort)

    results = {
        "experiment": "09_bayesian_vs_amortized",
        "config": {
            "n_stations": conf["S"], "quick": conf["quick"], "seed": conf["seed"],
            "mh_iter": conf["mh_iter"], "mh_burn": conf["mh_burn"],
            "amortized_train_sims": conf["train_sims"],
            "device": str(device),
            "generative": {"mu": params.mu, "tau": params.tau, "sigma": params.sigma},
        },
        "before_closed_form_known_hyper": metrics_conj,
        "before_mh": {
            **metrics_mh,
            "mh_accept_rate": mh.accept_rate,
            "post_hyper": {
                "mu_mean": float(mh.mu.mean()),
                "tau_mean": float(mh.tau.mean()),
                "sigma_mean": float(mh.sigma.mean()),
                "mu_95": [float(np.quantile(mh.mu, 0.025)),
                          float(np.quantile(mh.mu, 0.975))],
                "tau_95": [float(np.quantile(mh.tau, 0.025)),
                           float(np.quantile(mh.tau, 0.975))],
                "sigma_95": [float(np.quantile(mh.sigma, 0.025)),
                             float(np.quantile(mh.sigma, 0.975))],
            },
        },
        "after_amortized": metrics_amort,
        "wall_time_sec": {
            "before_closed_form": round(t_conj, 4),
            "before_mh_fit": round(t_mh, 4),
            "after_amortized_train_once": round(t_train, 4),
            "after_amortized_score": round(t_score, 4),
        },
        "speedup_score_vs_mh_fit": round(t_mh / max(t_score, 1e-9), 2),
    }

    # ----- Artifacts --------------------------------------------------------- #
    _plot_intervals(post_conj, post_mh, post_amort, theta_true)
    _plot_bars(results)
    _write_metrics_json(results)
    _write_summary_md(results)
    return results


# --------------------------------------------------------------------------- #
def _plot_intervals(post_conj, post_mh, post_amort, theta_true):
    plt = plotting.plt
    fig, ax = plotting.new_fig(10.0, 4.6)
    S = len(theta_true)
    x = np.arange(S)
    off = 0.22
    # MH and amortized (both targeted at the FULL posterior; conjugate is reference)
    for arr, name, color, dx in [
        (post_mh, "Before: MCMC (MH)", plotting.BEFORE_COLOR, -off),
        (post_amort, "After: amortized (MLP)", plotting.AFTER_COLOR, +off),
    ]:
        lo = arr["lo95"].to_numpy(); hi = arr["hi95"].to_numpy()
        pm = arr["post_mean"].to_numpy()
        ax.vlines(x + dx, lo, hi, color=color, lw=1.5, alpha=0.85, label=None)
        ax.plot(x + dx, pm, "o", color=color, ms=3.5, label=name)
    ax.plot(x, theta_true, "x", color="#202124", ms=6, mew=1.6, label="True theta_s")
    ax.set_xlabel("Station")
    ax.set_ylabel("Per-station effect (with 95% CI)")
    ax.set_title("Per-station posterior intervals: MCMC vs amortized vs truth")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plotting.save(fig, RESULTS_DIR / "posterior_intervals.png")


def _plot_bars(results: dict):
    """Two grouped bar charts: coverage and wall-time."""
    plt = plotting.plt
    mh = results["before_mh"]
    am = results["after_amortized"]
    wt = results["wall_time_sec"]

    # (a) coverage panel: nominal vs achieved + RMSE-to-truth
    labels = ["Coverage (95%)", "RMSE-to-truth"]
    before_vals = [mh["coverage_95"], mh["rmse_to_truth"]]
    after_vals = [am["coverage_95"], am["rmse_to_truth"]]
    # Plot coverage as percentage and RMSE as raw; chart two side-by-side panels.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    w = 0.38
    # Coverage panel
    ax = axes[0]
    ax.bar(-w/2, mh["coverage_95"], w, color=plotting.BEFORE_COLOR, label="Before: MCMC")
    ax.bar(+w/2, am["coverage_95"], w, color=plotting.AFTER_COLOR, label="After: amortized")
    ax.axhline(0.95, color="#d93025", lw=1.2, ls="--", label="Nominal 95%")
    ax.set_xticks([0]); ax.set_xticklabels(["Empirical coverage of 95% CI"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage (higher = closer to nominal)")
    ax.set_title("Coverage parity (MCMC vs amortized)")
    ax.legend(frameon=False)

    # Wall-time panel
    ax = axes[1]
    keys = ["MH fit", "Amortized score", "Amortized train (one-time)"]
    vals = [wt["before_mh_fit"], wt["after_amortized_score"], wt["after_amortized_train_once"]]
    colors = [plotting.BEFORE_COLOR, plotting.AFTER_COLOR, "#188038"]
    ax.bar(np.arange(3), vals, color=colors)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(keys, rotation=10, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Wall time (s, log)")
    ax.set_title("Per-dataset cost: MCMC vs amortized scoring")

    plt.tight_layout()
    plotting.save(fig, RESULTS_DIR / "before_after_bars.png")


def _write_metrics_json(results: dict) -> str:
    path = RESULTS_DIR / "metrics.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return str(path)


def _write_summary_md(results: dict) -> str:
    cfg = results["config"]
    cj = results["before_closed_form_known_hyper"]
    mh = results["before_mh"]
    am = results["after_amortized"]
    wt = results["wall_time_sec"]
    lines = [
        "# Experiment 09 — Bayesian hierarchical vs amortized inference: BEFORE vs AFTER",
        "",
        f"- Device: `{cfg['device']}`  |  stations S = {cfg['n_stations']}  "
        f"|  generative (mu, tau, sigma) = ({cfg['generative']['mu']}, "
        f"{cfg['generative']['tau']}, {cfg['generative']['sigma']})",
        f"- MH iters: {cfg['mh_iter']} (burn {cfg['mh_burn']})   "
        f"|  Amortized training datasets: {cfg['amortized_train_sims']}   "
        f"|  Quick: {cfg['quick']}",
        "",
        "## Per-station posterior quality vs the TRUE theta_s",
        "",
        "| Method | Coverage (95%) ~ 0.95 | RMSE-to-truth ↓ | Mean interval width ↓ |",
        "|--------|------:|------:|------:|",
        f"| Before: closed-form (known hyper) | {cj['coverage_95']:.3f} | "
        f"{cj['rmse_to_truth']:.3f} | {cj['mean_interval_width']:.3f} |",
        f"| Before: MCMC (MH, full posterior) | {mh['coverage_95']:.3f} | "
        f"{mh['rmse_to_truth']:.3f} | {mh['mean_interval_width']:.3f} |",
        f"| **After: amortized (MLP)** | **{am['coverage_95']:.3f}** | "
        f"**{am['rmse_to_truth']:.3f}** | **{am['mean_interval_width']:.3f}** |",
        "",
        "Coverage is the fraction of stations whose true effect falls inside the "
        "reported 95% CI; both Bayesian methods aim for ~0.95 (nominal). RMSE-to-truth "
        "summarizes how close the *posterior means* are to the latent truth (lower is "
        "sharper for the same coverage).",
        "",
        "## MCMC posterior over hyperparameters (informative diagnostics)",
        "",
        f"- mu ~ {mh['post_hyper']['mu_mean']:.3f}  "
        f"(95% CI {mh['post_hyper']['mu_95'][0]:.3f}..{mh['post_hyper']['mu_95'][1]:.3f}, "
        f"true {cfg['generative']['mu']})",
        f"- tau ~ {mh['post_hyper']['tau_mean']:.3f}  "
        f"(95% CI {mh['post_hyper']['tau_95'][0]:.3f}..{mh['post_hyper']['tau_95'][1]:.3f}, "
        f"true {cfg['generative']['tau']})",
        f"- sigma ~ {mh['post_hyper']['sigma_mean']:.3f}  "
        f"(95% CI {mh['post_hyper']['sigma_95'][0]:.3f}..{mh['post_hyper']['sigma_95'][1]:.3f}, "
        f"true {cfg['generative']['sigma']})",
        f"- MH acceptance rates: {mh['mh_accept_rate']}",
        "",
        "## Wall-time honest accounting (this machine)",
        "",
        "| Step | Time (s) |",
        "|------|---------:|",
        f"| BEFORE closed-form (known hyper) | {wt['before_closed_form']:.4f} |",
        f"| BEFORE MCMC (MH) fit             | {wt['before_mh_fit']:.4f} |",
        f"| AFTER amortized TRAIN (one-time) | {wt['after_amortized_train_once']:.4f} |",
        f"| AFTER amortized SCORE (per new dataset) | {wt['after_amortized_score']:.4f} |",
        "",
        f"**Speedup of amortized SCORING over MH FIT on a fresh dataset: "
        f"~{results['speedup_score_vs_mh_fit']:.1f}x.** Honest caveat: that one-time "
        "training cost is amortized over *all* future datasets from the same model "
        "family; if the model class changes you must retrain.",
        "",
        "Artifacts: `posterior_intervals.png`, `before_after_bars.png`, `metrics.json`.",
        "",
    ]
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    results = run(args)
    cj = results["before_closed_form_known_hyper"]
    mh = results["before_mh"]
    am = results["after_amortized"]
    wt = results["wall_time_sec"]
    cfg = results["config"]
    print(f"[exp09] device={cfg['device']} S={cfg['n_stations']} quick={cfg['quick']}")
    print(f"  BEFORE closed-form : coverage={cj['coverage_95']:.3f}  "
          f"RMSE={cj['rmse_to_truth']:.3f}  width={cj['mean_interval_width']:.3f}  "
          f"time={wt['before_closed_form']:.4f}s")
    print(f"  BEFORE MCMC (MH)   : coverage={mh['coverage_95']:.3f}  "
          f"RMSE={mh['rmse_to_truth']:.3f}  width={mh['mean_interval_width']:.3f}  "
          f"time={wt['before_mh_fit']:.4f}s")
    print(f"  AFTER  amortized   : coverage={am['coverage_95']:.3f}  "
          f"RMSE={am['rmse_to_truth']:.3f}  width={am['mean_interval_width']:.3f}  "
          f"train_once={wt['after_amortized_train_once']:.3f}s "
          f"score={wt['after_amortized_score']:.4f}s")
    print(f"  speedup score-vs-MH-fit: {results['speedup_score_vs_mh_fit']:.1f}x")
    print(f"[exp09] artifacts written to {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    main()
