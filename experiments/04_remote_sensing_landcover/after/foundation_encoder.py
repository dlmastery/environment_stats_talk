"""AFTER (foundation-model route) — frozen Prithvi-EO encoder + linear/kNN head.

This is the third leg of Experiment 04's before/after triad on HARD-mode data:

    - BEFORE              : RandomForest on hand-computed NDVI/NDWI/SWIR/brightness
    - AFTER (from-scratch): the SmallCNN trained from random init on the raw cube
    - AFTER (foundation)  : a *pretrained* remote-sensing encoder — **NASA-IBM
                            Prithvi-EO-1.0-100M** (Hugging Face id
                            ``ibm-nasa-geospatial/Prithvi-EO-1.0-100M``, weights
                            licensed Apache-2.0) — used FROZEN as a feature
                            extractor, with cheap linear / kNN heads trained on
                            the resulting CLS embeddings.

Why Prithvi (and not the other two candidates in the brief).
----------------------------------------------------------------
The brief asks us to try, in order, (1) EagleVision (arXiv:2503.23330), (2)
NASA-IBM Prithvi-EO-1.0-100M (HLS-pretrained ViT), and (3) a generic ImageNet
``timm`` ViT used on a 3-band projection of our patches.

We chose **Prithvi-EO-1.0-100M** because it is the closest *real RS-specific*
foundation encoder available on Hugging Face: a 100 M-parameter ViT
(``patch_size=16``, ``embed_dim=768``, ``depth=12``, ``num_heads=12``) pretrained
with a Masked Autoencoder on NASA's Harmonized Landsat-Sentinel-2 (HLS) product,
on six bands ``[Blue, Green, Red, Narrow-NIR, SWIR1, SWIR2]`` — directly
matching the spectral domain of our 5-band synthetic chips
``[blue, green, red, nir, swir]`` (Prithvi has one extra SWIR; we map our single
SWIR to *both* SWIR1 and SWIR2, which is a faithful, documented stand-in given
our generator doesn't model two SWIR sub-bands).

EagleVision is a remote-sensing **MLLM** built around a captioning/QA interface
(no clean pretrained embedding endpoint released as a turn-key HF model). Using
it as a frozen patch encoder for a tabular linear probe is not a fair "as
released" use of the model, so we documented it and moved to option (2). The
generic ``timm`` ImageNet fallback is implemented below as ``encode_timm_vit``
for transparency, but is **not** the headline foundation result — Prithvi is.

What the wrapper does, exactly
------------------------------
1. **Bands.** Our cube is ``(N, 5, H, W)`` with order
   ``[blue, green, red, nir, swir]``. We re-stack to 6 bands by repeating SWIR
   for SWIR1 and SWIR2: ``[blue, green, red, nir, swir, swir]``.
2. **Resolution.** Prithvi expects ``224x224`` patches; our HARD-mode patches
   are ``16x16`` (or whatever ``--size`` produced them). We bilinearly upsample
   to ``224x224``. (Bilinear upsampling does not invent texture but lets the ViT
   tile the patch with its ``16x16`` patch-embed.)
3. **Reflectance scale.** Our synthetic generator emits reflectance in ``[0, 1]``;
   Prithvi was trained on HLS surface-reflectance scaled to *thousands*
   (``data_mean`` ~ 700–2500). We multiply by 10 000 before applying Prithvi's
   per-band ``(mean, std)`` normalisation, so the input distribution matches.
4. **Embedding.** We call ``model.forward_features(x.unsqueeze(2))`` (adding the
   temporal dimension for a single frame), take the CLS token from the last
   block: ``feats[-1][:, 0, :]`` → a fixed ``768``-D vector per patch.
5. **Heads.** Two cheap, supervised heads trained on the TRAIN embeddings and
   evaluated on TEST: a **linear** ``LogisticRegression`` (multinomial linear
   probe — the standard foundation-model evaluation protocol) and a
   ``KNeighborsClassifier(k=5)`` (no fit, just nearest-neighbour transfer).

Honest-result discipline
------------------------
Everything below returns *what actually happened* on the synthetic HARD-mode
data. A frozen Prithvi + linear probe may or may not beat the from-scratch
SmallCNN — this synthetic texture-pair task is *not* in Prithvi's pretraining
distribution, so it is entirely possible that a 100 M ViT trained on real HLS
scenes is dominated by a tiny CNN purpose-built for the synthetic textures. We
report the real numbers and let the audience read them.

If the Prithvi weights cannot be downloaded (offline, SSL block, etc.) the
``load_prithvi_encoder`` function raises a clear ``FoundationEncoderUnavailable``
exception, and the orchestrator falls back to either ``encode_timm_vit`` (an
ImageNet ViT used as a generic feature extractor on a 3-band RGB projection of
our patches) or returns ``{"skipped": True, "reason": ...}`` — no fabricated
numbers are ever written to disk.

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common import metrics, BANDS  # noqa: E402

__all__ = [
    "FoundationEncoderUnavailable",
    "PrithviAdapter",
    "load_prithvi_encoder",
    "encode_timm_vit",
    "extract_embeddings",
    "evaluate",
]

# --------------------------------------------------------------------------- #
# Prithvi constants (verbatim from the model's published config.yaml)          #
# --------------------------------------------------------------------------- #
# Source: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
# config.yaml -> train_params.data_mean / data_std, for HLS L30 bands
# [B02, B03, B04, B05, B06, B07] == [Blue, Green, Red, Narrow-NIR, SWIR1, SWIR2].
PRITHVI_HF_ID = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]
PRITHVI_DATA_MEAN = np.array(
    [775.2290211032589, 1080.992780391705, 1228.5855250417867,
     2497.2022620507532, 2204.2139147975554, 1610.8324823273745],
    dtype=np.float32,
)
PRITHVI_DATA_STD = np.array(
    [1281.526139861424, 1270.0297974547493, 1399.4802505642526,
     1368.3446143747644, 1291.6764008585435, 1154.505683480695],
    dtype=np.float32,
)
PRITHVI_INPUT_SIZE = 224
PRITHVI_EMBED_DIM = 768


class FoundationEncoderUnavailable(RuntimeError):
    """Raised when the chosen foundation encoder cannot be loaded.

    The orchestrator catches this and either falls back to ``encode_timm_vit``
    (an ImageNet ViT — *generic*, not RS-specific, documented) or skips the run
    with a clear reason. We never write a fabricated metric to disk.
    """


# --------------------------------------------------------------------------- #
# Prithvi: load + adapt our 5-band [0,1] HARD-mode patches to its 6-band ViT.  #
# --------------------------------------------------------------------------- #
def _import_prithvi_module(prithvi_dir: Path):
    """Import ``prithvi_mae.py`` from the model's HF snapshot dir by file path.

    The repo ships the model definition as a top-level Python file (no package
    install). We load it via ``importlib.util.spec_from_file_location`` so we do
    not need to extend ``sys.path`` permanently.
    """
    src = prithvi_dir / "prithvi_mae.py"
    if not src.exists():
        raise FoundationEncoderUnavailable(
            f"prithvi_mae.py not found at {src} — did the HF snapshot download fully?"
        )
    # prithvi_mae.py itself imports from a few stdlib + torch + einops + timm
    # modules — all of which we already have. We must also keep its dir on
    # sys.path so any sibling imports inside the file resolve.
    if str(prithvi_dir) not in sys.path:
        sys.path.insert(0, str(prithvi_dir))
    spec = importlib.util.spec_from_file_location("prithvi_mae", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_prithvi_encoder(device: Optional[torch.device] = None):
    """Download + load Prithvi-EO-1.0-100M as a frozen ViT encoder (CLS embeddings).

    Returns ``(encoder, prithvi_dir)`` — the encoder is in ``.eval()`` mode with
    every parameter frozen; ``prithvi_dir`` is the HF snapshot directory (handy
    for inspection/debugging). Raises ``FoundationEncoderUnavailable`` if the
    download or load fails (we do *not* fabricate a result).
    """
    try:
        # Windows: TLS trust store; required before any HF download.
        import truststore  # type: ignore
        try:
            truststore.inject_into_ssl()
        except Exception:  # pragma: no cover - already injected
            pass
    except ImportError:  # pragma: no cover - optional on Linux
        pass

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise FoundationEncoderUnavailable(
            "huggingface_hub not installed; pip install huggingface_hub"
        ) from e

    try:
        ckpt_path = hf_hub_download(PRITHVI_HF_ID, "Prithvi_EO_V1_100M.pt")
        hf_hub_download(PRITHVI_HF_ID, "prithvi_mae.py")
        hf_hub_download(PRITHVI_HF_ID, "config.yaml")
    except Exception as e:
        raise FoundationEncoderUnavailable(
            f"Could not download Prithvi-EO-1.0-100M from Hugging Face: {e!r}"
        ) from e

    prithvi_dir = Path(ckpt_path).parent
    try:
        prithvi_mod = _import_prithvi_module(prithvi_dir)
    except Exception as e:
        raise FoundationEncoderUnavailable(
            f"Could not import prithvi_mae module: {e!r}"
        ) from e

    # Build PrithviViT (encoder only). num_frames=1: we have static patches, not
    # 3-frame HLS time-series; the patch_size temporal dim stays 1, so the
    # pretrained Conv3d patch-embed weights ((768, 6, 1, 16, 16)) load directly.
    PrithviViT = prithvi_mod.PrithviViT
    model = PrithviViT(
        img_size=PRITHVI_INPUT_SIZE,
        patch_size=(1, 16, 16),
        num_frames=1,
        in_chans=6,
        embed_dim=PRITHVI_EMBED_DIM,
        depth=12,
        num_heads=12,
        encoder_only=True,
    )

    # Load checkpoint. The published file is the FULL MAE; we keep encoder.*
    # weights only, strip the "encoder." prefix, and drop pos_embed (it is
    # regenerated on-the-fly inside ``forward_features`` for any input grid).
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    enc_sd = {}
    for k, v in sd.items():
        if k.startswith("encoder.") and "pos_embed" not in k:
            enc_sd[k[len("encoder."):]] = v
    res = model.load_state_dict(enc_sd, strict=False)
    # Tolerate exactly the 'pos_embed' missing key (regenerated dynamically); fail
    # loudly on anything else so we never silently use a half-initialised model.
    unexpected = list(res.unexpected_keys)
    missing = [k for k in res.missing_keys if k != "pos_embed"]
    if missing or unexpected:
        raise FoundationEncoderUnavailable(
            f"Prithvi checkpoint did not load cleanly: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )

    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, prithvi_dir


@dataclass
class PrithviAdapter:
    """Glue between our HARD-mode patches and Prithvi's expected input.

    Bands
    -----
    Our cube is ``(N, 5, H, W)`` in order ``[blue, green, red, nir, swir]``.
    Prithvi expects 6 HLS bands ``[B02=Blue, B03=Green, B04=Red, B05=Narrow-NIR,
    B06=SWIR1, B07=SWIR2]``. We re-stack to::

        [blue, green, red, nir, swir, swir]

    i.e. duplicate SWIR so it serves as both SWIR1 and SWIR2. The synthetic
    generator does not model two SWIR sub-bands, so this is a faithful,
    documented stand-in — and crucially, it lets us run a *real* pretrained
    encoder on our cubes without any fine-tuning.

    Reflectance scale
    -----------------
    Our generator emits reflectance in ``[0, 1]``; Prithvi was trained on HLS
    surface reflectance scaled to thousands (``data_mean`` ~ 700–2500). We
    multiply by 10 000 (the canonical HLS / Sentinel-2 L2A scale factor) before
    applying Prithvi's per-band ``(mean, std)`` normalisation so the input
    statistics match the pretraining distribution.

    Resolution
    ----------
    Prithvi expects 224×224 chips; our HARD-mode patches are 16×16. We bilinearly
    upsample (no smoothing of class structure beyond what bilinear interp does).
    """

    model: nn.Module
    device: torch.device
    img_size: int = PRITHVI_INPUT_SIZE
    batch_size: int = 32

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Run frozen Prithvi on (N,5,H,W) reflectance patches → (N,768) CLS embeddings."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[1] != len(BANDS):
            raise ValueError(
                f"expected (N, {len(BANDS)}, H, W), got {X.shape}"
            )
        # Reorder to 6 bands: [blue, green, red, nir, swir, swir]. BANDS is
        # already in that order, so this is bands 0..3 then swir(4) twice.
        x6 = np.stack(
            [X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 4]], axis=1
        )  # (N, 6, H, W)

        # Scale to HLS reflectance units (×10000) then per-band z-score.
        x6 = x6 * 10000.0
        mean = PRITHVI_DATA_MEAN.reshape(1, 6, 1, 1)
        std = PRITHVI_DATA_STD.reshape(1, 6, 1, 1)
        x6 = (x6 - mean) / std
        x6 = x6.astype(np.float32)

        embs = []
        N = x6.shape[0]
        for i in range(0, N, self.batch_size):
            xb = torch.from_numpy(x6[i:i + self.batch_size]).to(self.device)
            # bilinear upsample 16x16 -> 224x224 (no-op if already 224)
            if xb.shape[-1] != self.img_size or xb.shape[-2] != self.img_size:
                xb = nn.functional.interpolate(
                    xb, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False,
                )
            # add temporal dim -> (B, 6, 1, 224, 224); the model auto-adds it
            # in forward_features, but we do it explicitly to be unambiguous.
            xb = xb.unsqueeze(2)
            feats = self.model.forward_features(xb)
            cls = feats[-1][:, 0, :]  # (B, 768)
            embs.append(cls.detach().float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# --------------------------------------------------------------------------- #
# Generic timm fallback (ImageNet ViT — NOT RS-specific, documented)           #
# --------------------------------------------------------------------------- #
def encode_timm_vit(
    X: np.ndarray, *,
    model_name: str = "vit_small_patch16_224",
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, str]:
    """Embed (N,5,H,W) patches with a generic ImageNet ViT (3-band projection).

    Fallback path requested by the brief (option 3). This is NOT RS-specific —
    it is a real pretrained foundation encoder (ViT-Small, patch=16, 224×224,
    ImageNet-21k → 1k via ``timm``) used as a feature extractor on a 3-band
    RGB-like projection ``[red, green, blue]`` of our cube. We report it
    transparently as "generic ImageNet ViT" so the audience doesn't confuse it
    with the RS-pretrained Prithvi number.

    Returns ``(embeddings, encoder_label)``.
    """
    import timm  # imported lazily so a missing install raises cleanly

    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    X = np.asarray(X, dtype=np.float32)
    # Project our 5 bands onto an RGB-like 3-channel image: take [red, green, blue]
    # (indices 2,1,0 in our BANDS order) and stretch to [0,1] already.
    rgb = np.stack([X[:, 2], X[:, 1], X[:, 0]], axis=1)  # (N, 3, H, W)

    # ImageNet normalisation (the canonical mean/std for timm/torchvision ViTs).
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    rgb = (rgb - mean) / std

    embs = []
    with torch.no_grad():
        for i in range(0, rgb.shape[0], batch_size):
            xb = torch.from_numpy(rgb[i:i + batch_size]).to(device)
            xb = nn.functional.interpolate(
                xb, size=(224, 224), mode="bilinear", align_corners=False,
            )
            e = model(xb)  # (B, D) since num_classes=0
            embs.append(e.detach().float().cpu().numpy())
    return np.concatenate(embs, axis=0), f"timm/{model_name}"


# --------------------------------------------------------------------------- #
# Convenience: extract embeddings + train + evaluate linear & kNN heads.       #
# --------------------------------------------------------------------------- #
def extract_embeddings(
    X: np.ndarray, *,
    encoder: str = "prithvi",
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, str]:
    """Encode (N,5,H,W) patches with the requested foundation encoder.

    ``encoder`` is one of:
      - ``"prithvi"`` — NASA-IBM Prithvi-EO-1.0-100M (RS-pretrained, recommended).
      - ``"timm"``    — generic ImageNet ``vit_small_patch16_224`` (fallback).

    Returns ``(embeddings, encoder_label)`` for downstream heads.
    """
    if encoder == "prithvi":
        model, _ = load_prithvi_encoder(device=device)
        adapter = PrithviAdapter(model=model, device=next(model.parameters()).device)
        return adapter.embed(X), f"hf/{PRITHVI_HF_ID}"
    if encoder == "timm":
        return encode_timm_vit(X, device=device)
    raise ValueError(f"unknown encoder: {encoder!r}")


def _fit_heads(
    emb_tr: np.ndarray, y_tr: np.ndarray,
    emb_te: np.ndarray, y_te: np.ndarray,
    k_neighbors: int = 5, seed: int = 0,
) -> dict:
    """Fit + score a multinomial-LogReg linear probe AND a kNN head on embeddings.

    Both are cheap CPU-side classifiers. The linear probe is the standard
    foundation-model evaluation protocol; kNN is included because it requires
    *no training at all* (a stricter test of the embedding's separability).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    # Standardise (linear probes are scale-sensitive); fit on TRAIN only.
    scaler = StandardScaler().fit(emb_tr)
    Ztr = scaler.transform(emb_tr)
    Zte = scaler.transform(emb_te)

    out = {}
    # Linear probe — lbfgs multinomial loss; modest reg.
    # (sklearn>=1.7 dropped the deprecated ``multi_class`` kwarg; the default
    # solver/loss already does multinomial for >2 classes.)
    lp = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", random_state=seed,
    )
    lp.fit(Ztr, y_tr)
    y_pred_lp = lp.predict(Zte).astype(int)
    rep_lp = metrics.classification_report_simple(y_te, y_pred_lp)
    out["linear_probe"] = {
        "head": "LogisticRegression (multinomial linear probe)",
        "accuracy": rep_lp["accuracy"],
        "macro_f1": rep_lp["macro_f1"],
        "macro_precision": rep_lp["macro_precision"],
        "macro_recall": rep_lp["macro_recall"],
        "y_pred": y_pred_lp.tolist(),
    }

    # kNN head — no training, just nearest-neighbour transfer.
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(Ztr, y_tr)
    y_pred_knn = knn.predict(Zte).astype(int)
    rep_knn = metrics.classification_report_simple(y_te, y_pred_knn)
    out["knn"] = {
        "head": f"KNeighborsClassifier(k={k_neighbors})",
        "accuracy": rep_knn["accuracy"],
        "macro_f1": rep_knn["macro_f1"],
        "macro_precision": rep_knn["macro_precision"],
        "macro_recall": rep_knn["macro_recall"],
        "y_pred": y_pred_knn.tolist(),
    }
    return out


def evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, *,
    encoder: str = "prithvi",
    device: Optional[torch.device] = None,
    k_neighbors: int = 5,
    seed: int = 0,
) -> dict:
    """End-to-end: frozen encoder → (linear probe + kNN) heads on HARD-mode data.

    Returns a metrics dict with the encoder id, embedding dimension, and both
    heads' accuracy / macro-F1 / macro-precision / macro-recall — plus whether
    the run was skipped (with a reason) if the encoder could not load.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    try:
        emb_tr, label = extract_embeddings(X_train, encoder=encoder, device=device)
        emb_te, _ = extract_embeddings(X_test, encoder=encoder, device=device)
    except FoundationEncoderUnavailable as e:
        return {
            "skipped": True,
            "encoder": encoder,
            "reason": str(e),
        }

    heads = _fit_heads(
        emb_tr, np.asarray(y_train, int),
        emb_te, np.asarray(y_test, int),
        k_neighbors=k_neighbors, seed=seed,
    )
    return {
        "skipped": False,
        "encoder": label,
        "encoder_arch": "ViT-B/16 (Prithvi-EO-1.0-100M)" if encoder == "prithvi"
                        else "ViT-S/16 (ImageNet)",
        "rs_specific": encoder == "prithvi",
        "embedding_dim": int(emb_tr.shape[1]),
        "device": str(device),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "linear_probe": heads["linear_probe"],
        "knn": heads["knn"],
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import multispectral_patches_hard
    Xtr, ytr = multispectral_patches_hard(n=400, size=16, seed=0)
    Xte, yte = multispectral_patches_hard(n=200, size=16, seed=1)
    out = evaluate(Xtr, ytr, Xte, yte, encoder="prithvi")
    print(out)
