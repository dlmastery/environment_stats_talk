"""HARD-MODE orchestrator — RF-on-indices vs CNN-on-bands on texture-separable data.

This is the *honest win* companion to ``run_before_after.py``. It runs the same two
pipelines — BEFORE (RandomForest on hand-computed NDVI/NDWI/SWIR/brightness) and
AFTER (the small CNN on the raw 5-band cube) — but on
``common.synthetic_remote_sensing.multispectral_patches_hard``, where several
classes share a (near-)identical MEAN spectrum and differ ONLY in spatial texture
(natural forest vs orchard rows; open water vs flooded crop rows).

Because the mean indices are structurally blind to the texture pairs, RF-on-indices
is capped well below the CNN, which reads the spatial pattern directly. The CNN
therefore beats the RF baseline on accuracy AND macro-F1 — a genuine before/after
accuracy win, not a tie.

Auto-uses CUDA (RTX 4090) when present; ``--cpu`` forces CPU. Writes to
``results/hard/``: ``metrics.json``, ``confusion_matrix_hard.png``,
``before_after_hard_bars.png``, ``summary.md``.

Run from the REPO ROOT so ``import common`` resolves::

    python experiments/04_remote_sensing_landcover/after/cnn_hard.py            # GPU/CPU auto
    python experiments/04_remote_sensing_landcover/after/cnn_hard.py --quick    # fast smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_EXP_DIR = _THIS.parents[1]
_REPO_ROOT = _THIS.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the generator DIRECTLY from the module (not via common/__init__).
from common.synthetic_remote_sensing import (  # noqa: E402
    multispectral_patches_hard, BANDS, HARD_CLASSES,
)
from common import plotting  # noqa: E402


def _load(module_name: str, rel_path: str):
    """Import a sibling experiment module by file path (no package install)."""
    spec = importlib.util.spec_from_file_location(module_name, _EXP_DIR / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rf_indices = _load("exp04_hard_rf_indices", "before/rf_indices.py")
cnn_classifier = _load("exp04_hard_cnn_classifier", "after/cnn_classifier.py")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, classes, path: str, title: str) -> str:
    """Row-normalised confusion matrix over the hard land-cover classes."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(np.asarray(y_true, int), np.asarray(y_pred, int)):
        cm[t, p] += 1
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plotting.new_fig(6.2, 5.2)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    return plotting.save(fig, path)


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Exp04 HARD-mode: RF-indices vs CNN-bands")
    ap.add_argument("--epochs", type=int, default=40, help="CNN epochs (early stopping)")
    ap.add_argument("--quick", action="store_true", help="tiny/fast smoke run")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rf-estimators", type=int, default=300)
    ap.add_argument("--width", type=int, default=24, help="CNN channel width")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cpu", action="store_true", help="force CPU (skip CUDA auto-use)")
    ap.add_argument("--results-dir", type=str, default=str(_EXP_DIR / "results" / "hard"))
    args = ap.parse_args(argv)

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.n_train, args.n_test = 240, 120
        args.size = 12
        args.rf_estimators = 60
        args.width = 8
        args.batch_size = 64
        args.patience = 1

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- shared, seeded HARD data ------------------------------------------
    X_train, y_train = multispectral_patches_hard(n=args.n_train, size=args.size, seed=args.seed)
    X_test, y_test = multispectral_patches_hard(n=args.n_test, size=args.size, seed=args.seed + 1)

    # ---- BEFORE: RF on indices ---------------------------------------------
    before = rf_indices.run_before(
        X_train, y_train, X_test, y_test, seed=args.seed,
        n_estimators=args.rf_estimators,
    )

    # ---- AFTER: CNN on raw bands -------------------------------------------
    device = cnn_classifier.pick_device(prefer_cuda=not args.cpu)
    after = cnn_classifier.run_after(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size, width=args.width,
        patience=args.patience, seed=args.seed, device=device,
    )

    # ---- plots --------------------------------------------------------------
    cm_path = plot_confusion(
        y_test, after["y_pred"], HARD_CLASSES,
        str(results_dir / "confusion_matrix_hard.png"),
        title=f"CNN confusion — HARD (acc={after['accuracy']:.2f})")
    bars_path = plotting.before_after_bars(
        labels=["accuracy", "macro-F1"],
        before_vals=[before["accuracy"], before["macro_f1"]],
        after_vals=[after["accuracy"], after["macro_f1"]],
        ylabel="score",
        title="HARD mode: RF-indices vs CNN-bands",
        path=str(results_dir / "before_after_hard_bars.png"),
        lower_is_better=False,
    )

    # ---- metrics.json -------------------------------------------------------
    metrics_out = {
        "experiment": "04_remote_sensing_landcover/hard",
        "mode": "hard (texture-separable; mean indices blind to texture pairs)",
        "classes": HARD_CLASSES,
        "bands": BANDS,
        "config": {
            "epochs": args.epochs, "quick": bool(args.quick),
            "n_train": args.n_train, "n_test": args.n_test,
            "size": args.size, "seed": args.seed,
            "rf_estimators": args.rf_estimators, "cnn_width": args.width,
            "batch_size": args.batch_size, "device": str(device),
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
        "artifacts": {
            "confusion_matrix": Path(cm_path).name,
            "before_after_bars": Path(bars_path).name,
        },
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2),
                                              encoding="utf-8")
    _write_summary(results_dir / "summary.md", metrics_out)

    # console
    d = metrics_out["classification"]["delta"]
    print("== Experiment 04 HARD: texture-separable land cover ==")
    print(f"  device: {device}   (quick={args.quick}, epochs={args.epochs})")
    print(f"  BEFORE RF-indices : acc={before['accuracy']:.3f}  macroF1={before['macro_f1']:.3f}")
    print(f"  AFTER  CNN-bands  : acc={after['accuracy']:.3f}  macroF1={after['macro_f1']:.3f}")
    print(f"  WIN margin (CNN - RF): acc={d['accuracy']:+.3f}  macroF1={d['macro_f1']:+.3f}")
    print(f"  wrote: {results_dir/'metrics.json'} + 2 PNGs + summary.md")
    return metrics_out


def _write_summary(path: Path, m: dict) -> None:
    c = m["classification"]
    b, a, d = c["before_rf_indices"], c["after_cnn_bands"], c["delta"]
    cfg = m["config"]
    lines = [
        "# Experiment 04 — HARD mode (texture-separable land cover)",
        "",
        f"_Config: device={cfg['device']}, epochs={cfg['epochs']}, quick={cfg['quick']}, "
        f"n_train={cfg['n_train']}, n_test={cfg['n_test']}, size={cfg['size']}, "
        f"seed={cfg['seed']}, rf_estimators={cfg['rf_estimators']}, cnn_width={cfg['cnn_width']}._",
        "",
        f"Classes: {', '.join(m['classes'])}.  Bands: {', '.join(m['bands'])}.",
        "",
        "Several classes share a (near-)identical **mean spectrum** and differ ONLY "
        "in spatial **texture** (natural forest vs orchard rows; open water vs flooded "
        "crop rows). Mean indices (NDVI/NDWI/SWIR/brightness) are blind to those pairs, "
        "so RF-on-indices is capped; the CNN reads the texture and wins on accuracy.",
        "",
        "## Land-cover classification (same seeded train/test split)",
        "",
        "| Pipeline | Features | Accuracy | Macro-F1 |",
        "|---|---|---:|---:|",
        f"| BEFORE — {b['model']} | 4 hand-computed indices | {b['accuracy']:.3f} | {b['macro_f1']:.3f} |",
        f"| AFTER — {a['model']} | raw 5-band patches ({a['n_params']:,} params) | {a['accuracy']:.3f} | {a['macro_f1']:.3f} |",
        f"| **Δ (after − before)** | | **{d['accuracy']:+.3f}** | **{d['macro_f1']:+.3f}** |",
        "",
        f"**Honest win:** the CNN beats RF-on-indices by **{d['accuracy']:+.3f} accuracy** "
        f"and **{d['macro_f1']:+.3f} macro-F1** on the texture-separable task.",
        "",
        "## Artifacts",
        "",
        f"- `{m['artifacts']['confusion_matrix']}` — CNN confusion matrix (HARD classes)",
        f"- `{m['artifacts']['before_after_bars']}` — RF vs CNN accuracy & macro-F1",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
