"""Exp04 — pretrained foundation-model route on HARD-mode land cover.

This is the *third* leg of the Experiment 04 before/after triad, ADDITIVE to
the existing artefacts (it never touches ``results/hard/metrics.json`` or any
other committed file under ``results/``):

    - BEFORE              : RandomForest on hand-computed NDVI/NDWI/SWIR/brightness
                            (committed: ``results/hard/metrics.json``)
    - AFTER (from-scratch): SmallCNN trained from random init on the raw cube
                            (committed: ``results/hard/metrics.json``)
    - AFTER (foundation)  : NASA-IBM **Prithvi-EO-1.0-100M** used FROZEN as a
                            feature extractor + a tiny linear probe / kNN head.
                            ⇒ NEW, written to ``results/hard/foundation_metrics.json``
                              and ``results/hard/foundation_comparison.png``.

Run from the REPO ROOT so ``import common`` resolves::

    python experiments/04_remote_sensing_landcover/run_foundation.py            # GPU/CPU auto
    python experiments/04_remote_sensing_landcover/run_foundation.py --quick    # fast smoke
    python experiments/04_remote_sensing_landcover/run_foundation.py --encoder timm
        # fallback: generic ImageNet ViT-S/16 on a 3-band RGB projection
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_EXP_DIR = _THIS.parent
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

# Import generator DIRECTLY from the module (matches cnn_hard.py pattern).
from common.synthetic_remote_sensing import (  # noqa: E402
    multispectral_patches_hard, BANDS, HARD_CLASSES,
)
from common import plotting  # noqa: E402

# Local import of the wrapper.
sys.path.insert(0, str(_EXP_DIR / "after"))
import foundation_encoder as fe  # noqa: E402


def _load_existing_hard_metrics() -> dict | None:
    """Read the committed ``results/hard/metrics.json`` if present (additive only).

    We *read* the existing BEFORE (RF-indices) + AFTER (from-scratch CNN)
    numbers so the comparison chart shows all three pipelines on the same axes.
    We **never** modify or overwrite that file — the foundation run writes to a
    SEPARATE ``foundation_metrics.json``.
    """
    path = _EXP_DIR / "results" / "hard" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_three_way_bars(
    rf_acc: float | None, rf_f1: float | None,
    cnn_acc: float | None, cnn_f1: float | None,
    fnd_lp_acc: float, fnd_lp_f1: float,
    fnd_knn_acc: float, fnd_knn_f1: float,
    fnd_label: str,
    out_path: Path,
) -> str:
    """Side-by-side bar chart: RF-indices vs from-scratch CNN vs frozen-encoder heads.

    Bars whose source values are missing (e.g. running before the hard-mode
    committed metrics exist) are simply omitted — never hallucinated.
    """
    import matplotlib.pyplot as plt

    rows = []
    if rf_acc is not None:
        rows.append(("BEFORE: RF on indices", rf_acc, rf_f1, plotting.BEFORE_COLOR))
    if cnn_acc is not None:
        rows.append(("AFTER: from-scratch SmallCNN", cnn_acc, cnn_f1, plotting.AFTER_COLOR))
    rows.append((f"AFTER: {fnd_label} + linear probe", fnd_lp_acc, fnd_lp_f1, plotting.ACCENT))
    rows.append((f"AFTER: {fnd_label} + kNN(k=5)", fnd_knn_acc, fnd_knn_f1, "#7a4fcf"))

    labels = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    colors = [r[3] for r in rows]

    fig, ax = plotting.new_fig(9.5, 4.8)
    x = np.arange(len(labels))
    w = 0.38
    bars1 = ax.bar(x - w / 2, accs, w, label="accuracy", color=colors,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, f1s, w, label="macro-F1",
                   color=[plt.matplotlib.colors.to_rgba(c, 0.55) for c in colors],
                   edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Exp04 HARD: three-way comparison (real numbers, this machine)")
    ax.axhline(0.2, color="grey", lw=0.6, ls=":", alpha=0.6)
    ax.text(len(labels) - 0.5, 0.21, "5-class chance", color="grey", fontsize=8,
            ha="right", va="bottom")
    for bars, vals in ((bars1, accs), (bars2, f1s)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    return plotting.save(fig, str(out_path))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(
        description="Exp04 HARD: frozen foundation encoder + linear/kNN heads")
    ap.add_argument("--encoder", choices=["prithvi", "timm"], default="prithvi",
                    help="prithvi (RS-pretrained, recommended) or timm (generic ImageNet ViT-S/16)")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--size", type=int, default=16, help="HARD-mode patch size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--quick", action="store_true",
                    help="tiny/fast smoke run (n=120/60, 12x12 patches)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (default: CUDA if available)")
    ap.add_argument("--results-dir", type=str,
                    default=str(_EXP_DIR / "results" / "hard"))
    args = ap.parse_args(argv)

    if args.quick:
        args.n_train, args.n_test = 120, 60
        args.size = 12
        args.batch_size = 16

    device = (torch.device("cpu") if args.cpu
              else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- shared, seeded HARD data (same generator as cnn_hard.py) -----------
    t_data = time.time()
    X_train, y_train = multispectral_patches_hard(n=args.n_train, size=args.size, seed=args.seed)
    X_test, y_test = multispectral_patches_hard(n=args.n_test, size=args.size, seed=args.seed + 1)
    t_data = time.time() - t_data

    # ---- foundation encoder + linear/kNN ------------------------------------
    t_emb = time.time()
    # Manually wire so we can time embed-only vs head-only and reuse the same
    # encoder for TRAIN and TEST without reloading weights.
    try:
        if args.encoder == "prithvi":
            model, _ = fe.load_prithvi_encoder(device=device)
            adapter = fe.PrithviAdapter(model=model, device=device,
                                         batch_size=args.batch_size)
            emb_tr = adapter.embed(X_train)
            emb_te = adapter.embed(X_test)
            enc_label_full = f"hf/{fe.PRITHVI_HF_ID}"
            enc_label_short = "Prithvi-EO-100M (frozen)"
            arch_label = "ViT-B/16 (Prithvi-EO-1.0-100M, HLS-pretrained)"
            rs_specific = True
        else:
            emb_tr, enc_label_full = fe.encode_timm_vit(
                X_train, device=device, batch_size=args.batch_size)
            emb_te, _ = fe.encode_timm_vit(
                X_test, device=device, batch_size=args.batch_size)
            enc_label_short = "ImageNet ViT-S/16 (frozen)"
            arch_label = "ViT-S/16 (ImageNet, generic)"
            rs_specific = False
    except fe.FoundationEncoderUnavailable as e:
        out = {
            "skipped": True,
            "encoder": args.encoder,
            "reason": str(e),
            "config": {"n_train": args.n_train, "n_test": args.n_test,
                       "size": args.size, "seed": args.seed},
        }
        (results_dir / "foundation_metrics.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")
        print("[skip]", e)
        return out
    t_emb = time.time() - t_emb

    t_heads = time.time()
    heads = fe._fit_heads(
        emb_tr, np.asarray(y_train, int),
        emb_te, np.asarray(y_test, int),
        k_neighbors=args.k_neighbors, seed=args.seed,
    )
    t_heads = time.time() - t_heads

    # ---- pull RF + from-scratch CNN numbers for the three-way chart ---------
    existing = _load_existing_hard_metrics()
    rf_acc = rf_f1 = cnn_acc = cnn_f1 = None
    rf_model = cnn_model = None
    if existing is not None:
        c = existing.get("classification", {})
        b = c.get("before_rf_indices", {})
        a = c.get("after_cnn_bands", {})
        rf_acc, rf_f1 = b.get("accuracy"), b.get("macro_f1")
        cnn_acc, cnn_f1 = a.get("accuracy"), a.get("macro_f1")
        rf_model, cnn_model = b.get("model"), a.get("model")

    chart_path = _plot_three_way_bars(
        rf_acc, rf_f1, cnn_acc, cnn_f1,
        heads["linear_probe"]["accuracy"], heads["linear_probe"]["macro_f1"],
        heads["knn"]["accuracy"], heads["knn"]["macro_f1"],
        fnd_label=enc_label_short,
        out_path=results_dir / "foundation_comparison.png",
    )

    # ---- write foundation_metrics.json (ADDITIVE; never touches metrics.json) #
    metrics_out = {
        "experiment": "04_remote_sensing_landcover/hard/foundation",
        "mode": "hard (texture-separable; mean indices blind to texture pairs)",
        "encoder": enc_label_full,
        "encoder_arch": arch_label,
        "encoder_rs_specific": rs_specific,
        "encoder_frozen": True,
        "embedding_dim": int(emb_tr.shape[1]),
        "classes": HARD_CLASSES,
        "bands": BANDS,
        "config": {
            "n_train": args.n_train, "n_test": args.n_test,
            "size": args.size, "seed": args.seed,
            "batch_size": args.batch_size, "device": str(device),
            "quick": bool(args.quick),
            "k_neighbors": args.k_neighbors,
        },
        "timing_sec": {
            "data": round(t_data, 3),
            "embedding": round(t_emb, 3),
            "head_fit_predict": round(t_heads, 3),
        },
        "heads": {
            "linear_probe": heads["linear_probe"],
            "knn": heads["knn"],
        },
        "comparison_inputs_from_existing_metrics_json": {
            "before_rf_indices": (None if rf_acc is None else
                                  {"model": rf_model, "accuracy": rf_acc, "macro_f1": rf_f1}),
            "after_cnn_bands": (None if cnn_acc is None else
                                {"model": cnn_model, "accuracy": cnn_acc, "macro_f1": cnn_f1}),
        },
        "deltas_vs_existing": _compute_deltas(rf_acc, rf_f1, cnn_acc, cnn_f1, heads),
        "artifacts": {
            "comparison_chart": Path(chart_path).name,
        },
        "notes": [
            "Frozen encoder — zero parameters updated on this task.",
            "Heads (linear probe, kNN) trained on TRAIN embeddings only.",
            "5-band → 6-band: SWIR duplicated as SWIR1+SWIR2 (synthetic generator "
            "does not model two SWIR sub-bands).",
            "Patches bilinearly upsampled from {size}x{size} to 224x224 for Prithvi.".format(size=args.size),
            "Reflectance scaled ×10000 then normalised with Prithvi's HLS per-band "
            "(mean, std) from the published config.yaml.",
        ],
    }
    (results_dir / "foundation_metrics.json").write_text(
        json.dumps(metrics_out, indent=2), encoding="utf-8")

    # ---- console -----------------------------------------------------------
    print("== Experiment 04 HARD: foundation encoder + linear/kNN heads ==")
    print(f"  encoder      : {enc_label_full}  ({'RS-specific' if rs_specific else 'generic ImageNet'})")
    print(f"  device       : {device}    embedding_dim={emb_tr.shape[1]}")
    print(f"  n_train/test : {args.n_train}/{args.n_test}    patch={args.size}x{args.size}")
    print(f"  timing       : data={t_data:.1f}s  embed={t_emb:.1f}s  heads={t_heads:.2f}s")
    print(f"  LINEAR PROBE : acc={heads['linear_probe']['accuracy']:.3f}  "
          f"macroF1={heads['linear_probe']['macro_f1']:.3f}")
    print(f"  kNN(k={args.k_neighbors})     : acc={heads['knn']['accuracy']:.3f}  "
          f"macroF1={heads['knn']['macro_f1']:.3f}")
    if rf_acc is not None and cnn_acc is not None:
        d = metrics_out["deltas_vs_existing"]
        print(f"  delta vs RF  : LP acc {d['linear_probe_vs_rf']['accuracy']:+.3f}  "
              f"LP F1 {d['linear_probe_vs_rf']['macro_f1']:+.3f}")
        print(f"  delta vs CNN : LP acc {d['linear_probe_vs_cnn']['accuracy']:+.3f}  "
              f"LP F1 {d['linear_probe_vs_cnn']['macro_f1']:+.3f}")
    print(f"  wrote: {results_dir/'foundation_metrics.json'} + foundation_comparison.png")
    return metrics_out


def _compute_deltas(rf_acc, rf_f1, cnn_acc, cnn_f1, heads):
    """Compute (linear-probe − {RF, CNN}) and (kNN − {RF, CNN}) deltas honestly."""
    lp = heads["linear_probe"]
    knn = heads["knn"]
    out = {}
    if rf_acc is not None:
        out["linear_probe_vs_rf"] = {
            "accuracy": lp["accuracy"] - rf_acc, "macro_f1": lp["macro_f1"] - rf_f1}
        out["knn_vs_rf"] = {
            "accuracy": knn["accuracy"] - rf_acc, "macro_f1": knn["macro_f1"] - rf_f1}
    if cnn_acc is not None:
        out["linear_probe_vs_cnn"] = {
            "accuracy": lp["accuracy"] - cnn_acc, "macro_f1": lp["macro_f1"] - cnn_f1}
        out["knn_vs_cnn"] = {
            "accuracy": knn["accuracy"] - cnn_acc, "macro_f1": knn["macro_f1"] - cnn_f1}
    return out


if __name__ == "__main__":
    main()
