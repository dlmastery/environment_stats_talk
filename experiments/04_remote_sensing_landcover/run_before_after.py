"""Orchestrate Experiment 04: land-cover classification + change detection.

Runs the BEFORE pipeline (RandomForest on hand-computed spectral indices) and the
AFTER pipeline (a small PyTorch CNN on raw 5-band patches) on the *same* seeded
train/test split, plus an NDVI-difference change-detection mini-task, then writes:

    results/metrics.json          -- accuracy + macro-F1 for RF & CNN, change-detect F1
    results/confusion_matrix.png  -- CNN confusion matrix over the 5 classes
    results/change_map.png        -- before/after/NDVI-drop/change-mask for one pair
    results/before_after_bars.png -- RF vs CNN accuracy & macro-F1 bars
    results/summary.md            -- human-readable before/after summary table

Defaults are modest (finish in ~1-2 min on CPU). Use ``--quick`` for a fast smoke
run and ``--epochs N`` to control CNN training length. The full / headline run
(more data, more epochs, CUDA) is left for the GPU pass -- this script does not
force it.

Run from the REPO ROOT so ``import common`` resolves::

    python experiments/04_remote_sensing_landcover/run_before_after.py --quick
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_EXP_DIR = _THIS.parent
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import (  # noqa: E402
    multispectral_patches, change_pair, BANDS, CLASSES, metrics, plotting,
)
from common.plotting import plt  # noqa: E402


def _load(module_name: str, rel_path: str):
    """Import a sibling experiment module by file path (no package install)."""
    spec = importlib.util.spec_from_file_location(module_name, _EXP_DIR / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf_indices = _load("exp04_rf_indices", "before/rf_indices.py")
cnn_classifier = _load("exp04_cnn_classifier", "after/cnn_classifier.py")
change_detect = _load("exp04_change_detect", "after/change_detect.py")


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, path: str, title: str) -> str:
    """Render a normalised confusion matrix over the land-cover classes."""
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        cm[t, p] += 1
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row, 1, None)

    fig, ax = plotting.new_fig(5.6, 4.8)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    return plotting.save(fig, path)


def plot_change_map(X0, X1, change_res, path: str, pair_idx: int = 0) -> str:
    """Render before/after RGB, the NDVI-drop heatmap, and the change mask.

    The change *mask* is shown as a strip coloured by per-patch prediction (the
    synthetic patches are spatially near-uniform, so a per-patch map is the honest
    visualisation; on real chips this becomes a pixel-level change raster).
    """
    # RGB = red, green, blue bands (idx 2,1,0); stretch for display only.
    def rgb(X):
        img = np.stack([X[2], X[1], X[0]], axis=-1)
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    drop = np.asarray(change_res["ndvi_drop"], float)
    pred = np.asarray(change_res["y_pred"], int)
    true = np.asarray(change_res["y_true"], int)
    order = np.argsort(-drop)  # most-changed first for a readable strip

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    axes[0, 0].imshow(rgb(X0[pair_idx])); axes[0, 0].set_title("BEFORE (t0) RGB")
    axes[0, 1].imshow(rgb(X1[pair_idx])); axes[0, 1].set_title("AFTER (t1) RGB")
    ndvi_patch = (X1[pair_idx, 3] - X1[pair_idx, 2]) - (X0[pair_idx, 3] - X0[pair_idx, 2])
    im = axes[1, 0].imshow(-ndvi_patch, cmap="RdYlGn_r")
    axes[1, 0].set_title("NDVI drop (one pair)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    strip = np.stack([pred[order], true[order]], axis=0)
    axes[1, 1].imshow(strip, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                      interpolation="nearest")
    axes[1, 1].set_yticks([0, 1]); axes[1, 1].set_yticklabels(["predicted", "true"])
    axes[1, 1].set_xlabel("patches (sorted by NDVI drop)")
    axes[1, 1].set_title(f"change mask  (F1={change_res['macro_f1']:.2f})")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[1, 1].grid(False)
    fig.suptitle("Change detection: NDVI-difference threshold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return plotting.save(fig, path)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Experiment 04 before/after orchestrator")
    ap.add_argument("--epochs", type=int, default=25, help="CNN training epochs (with early stopping)")
    ap.add_argument("--quick", action="store_true", help="tiny/fast smoke run")
    ap.add_argument("--n-train", type=int, default=900)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--n-change", type=int, default=400)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rf-estimators", type=int, default=200)
    ap.add_argument("--cpu", action="store_true", help="force CPU (skip CUDA auto-use)")
    ap.add_argument("--results-dir", type=str, default=str(_EXP_DIR / "results"))
    args = ap.parse_args(argv)

    if args.quick:
        args.epochs = min(args.epochs, 3)
        args.n_train, args.n_test, args.n_change = 200, 100, 150
        args.size = 12
        args.rf_estimators = 80

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- shared, seeded data ------------------------------------------------
    X_train, y_train = multispectral_patches(n=args.n_train, size=args.size, seed=args.seed)
    X_test, y_test = multispectral_patches(n=args.n_test, size=args.size, seed=args.seed + 1)
    X0, X1, changed = change_pair(n=args.n_change, size=args.size, seed=args.seed + 2)

    # ---- BEFORE: RF on indices ---------------------------------------------
    before = rf_indices.run_before(
        X_train, y_train, X_test, y_test, seed=args.seed,
        n_estimators=args.rf_estimators,
    )

    # ---- AFTER: CNN on raw bands -------------------------------------------
    device = cnn_classifier.pick_device(prefer_cuda=not args.cpu)
    after = cnn_classifier.run_after(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs, seed=args.seed, device=device,
    )

    # ---- change detection ---------------------------------------------------
    change = change_detect.run_change(X0, X1, changed)

    # ---- plots --------------------------------------------------------------
    cm_path = plot_confusion(
        y_test, after["y_pred"], str(results_dir / "confusion_matrix.png"),
        title=f"CNN confusion (acc={after['accuracy']:.2f})")
    change_map_path = plot_change_map(X0, X1, change, str(results_dir / "change_map.png"))
    bars_path = plotting.before_after_bars(
        labels=["accuracy", "macro-F1"],
        before_vals=[before["accuracy"], before["macro_f1"]],
        after_vals=[after["accuracy"], after["macro_f1"]],
        ylabel="score", title="Land-cover classification: RF-indices vs CNN-bands",
        path=str(results_dir / "before_after_bars.png"), lower_is_better=False,
    )

    # ---- metrics.json -------------------------------------------------------
    metrics_out = {
        "experiment": "04_remote_sensing_landcover",
        "classes": CLASSES,
        "bands": BANDS,
        "config": {
            "epochs": args.epochs, "quick": bool(args.quick),
            "n_train": args.n_train, "n_test": args.n_test,
            "n_change": args.n_change, "size": args.size, "seed": args.seed,
            "rf_estimators": args.rf_estimators, "device": str(device),
        },
        "classification": {
            "before_rf_indices": {
                "model": before["model"],
                "accuracy": before["accuracy"],
                "macro_f1": before["macro_f1"],
                "feature_importance": before["feature_importance"],
            },
            "after_cnn_bands": {
                "model": after["model"],
                "device": after["device"],
                "accuracy": after["accuracy"],
                "macro_f1": after["macro_f1"],
                "n_params": after["n_params"],
                "best_epoch": after["best_epoch"],
                "epochs_run": after["epochs_run"],
            },
            "delta": {
                "accuracy": after["accuracy"] - before["accuracy"],
                "macro_f1": after["macro_f1"] - before["macro_f1"],
            },
        },
        "change_detection": {
            "model": change["task"],
            "threshold": change["threshold"],
            "accuracy": change["accuracy"],
            "macro_f1": change["macro_f1"],
            "n": change["n"],
        },
        "artifacts": {
            "confusion_matrix": Path(cm_path).name,
            "change_map": Path(change_map_path).name,
            "before_after_bars": Path(bars_path).name,
        },
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2),
                                               encoding="utf-8")

    # ---- summary.md ---------------------------------------------------------
    _write_summary(results_dir / "summary.md", metrics_out)

    # console
    print("== Experiment 04: land cover + change detection ==")
    print(f"  device: {device}   (quick={args.quick}, epochs={args.epochs})")
    print(f"  BEFORE RF-indices : acc={before['accuracy']:.3f}  macroF1={before['macro_f1']:.3f}")
    print(f"  AFTER  CNN-bands  : acc={after['accuracy']:.3f}  macroF1={after['macro_f1']:.3f}")
    print(f"  change detection  : acc={change['accuracy']:.3f}  macroF1={change['macro_f1']:.3f}  (thr={change['threshold']:.3f})")
    print(f"  wrote: {results_dir/'metrics.json'} and 3 PNGs + summary.md")
    return metrics_out


def _write_summary(path: Path, m: dict) -> None:
    c = m["classification"]
    b, a = c["before_rf_indices"], c["after_cnn_bands"]
    cd = m["change_detection"]
    cfg = m["config"]
    lines = [
        "# Experiment 04 — Land-cover classification + change detection",
        "",
        f"_Config: device={cfg['device']}, epochs={cfg['epochs']}, quick={cfg['quick']}, "
        f"n_train={cfg['n_train']}, n_test={cfg['n_test']}, seed={cfg['seed']}._",
        "",
        f"Classes: {', '.join(m['classes'])}.  Bands: {', '.join(m['bands'])}.",
        "",
        "## Land-cover classification (same seeded train/test split)",
        "",
        "| Pipeline | Features | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
        f"| BEFORE — {b['model']} | 4 hand-computed indices | {b['accuracy']:.3f} | {b['macro_f1']:.3f} |",
        f"| AFTER — {a['model']} | raw 5-band patches ({a['n_params']:,} params) | {a['accuracy']:.3f} | {a['macro_f1']:.3f} |",
        f"| **Δ (after − before)** | | **{c['delta']['accuracy']:+.3f}** | **{c['delta']['macro_f1']:+.3f}** |",
        "",
        "## Change detection (NDVI-difference threshold)",
        "",
        f"- threshold (calibrated for max macro-F1): **{cd['threshold']:.3f}**",
        f"- accuracy: **{cd['accuracy']:.3f}**, macro-F1: **{cd['macro_f1']:.3f}** (n={cd['n']})",
        "",
        "## Artifacts",
        "",
        f"- `{m['artifacts']['confusion_matrix']}` — CNN confusion matrix",
        f"- `{m['artifacts']['change_map']}` — before/after + NDVI-drop + change mask",
        f"- `{m['artifacts']['before_after_bars']}` — RF vs CNN accuracy & macro-F1",
        "",
        "_Note: numbers above are from whatever run produced them (e.g. `--quick`). "
        "The headline run (more data, more epochs, CUDA) is produced on the 4090._",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
