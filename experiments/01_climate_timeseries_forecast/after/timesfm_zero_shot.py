"""Real zero-shot foundation-model baseline for Exp01 using Google TimesFM.

Loads TimesFM (pytorch) from Hugging Face via the ``transformers`` library and
evaluates it zero-shot on the **same** synthetic temperature series and **same**
chronological split as the other Exp01 baselines, so the comparison is
apples-to-apples with Chronos. Uses CUDA when available.

This is the genuine "after / foundation model" path — no fine-tuning, just
``model(past_values, freq)`` and we take the model's mean point forecast.

Reference: Das et al. (2024), "A decoder-only foundation model for time-series
forecasting", arXiv:2310.10688. HF model: google/timesfm-2.0-500m-pytorch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Windows HF SSL fix: trust the OS truststore (matches chronos_zero_shot.py).
try:  # pragma: no cover - environmental shim
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:
    pass

# Make ``import common`` resolve when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics as env_metrics  # noqa: E402

DEFAULT_MODEL = "google/timesfm-2.0-500m-pytorch"
FALLBACK_MODEL = "google/timesfm-1.0-200m-pytorch"


def _load_model(prefer: str = DEFAULT_MODEL):
    """Load TimesFM via transformers. Falls back to 1.0-200m if 2.0 fails."""
    from transformers import TimesFmModelForPrediction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # fp32 is most numerically stable for forecasting
    try:
        model = TimesFmModelForPrediction.from_pretrained(prefer, torch_dtype=dtype)
        name = prefer
    except Exception as e:
        print(f"[timesfm] {prefer} unavailable ({e!r}); falling back to {FALLBACK_MODEL}.")
        model = TimesFmModelForPrediction.from_pretrained(FALLBACK_MODEL, torch_dtype=dtype)
        name = FALLBACK_MODEL
    model = model.to(device).eval()
    return model, name, device


def _predict_batch(model, contexts: list[np.ndarray], prediction_length: int) -> np.ndarray:
    """Return mean predictions of shape (B, prediction_length).

    The HF TimesFm model produces a fixed-length forecast (``horizon_length``,
    typically 128). We slice the first ``prediction_length`` steps. Inputs are
    right-padded to a common length (the model's context_length cap).
    """
    device = next(model.parameters()).device
    horizon = int(model.config.horizon_length)
    context_max = int(model.config.context_length)
    # Right-pad each context to the same length (use the last value to extend).
    # We cap at context_max; truncation keeps the most recent points.
    lens = [min(len(c), context_max) for c in contexts]
    L = max(lens)
    B = len(contexts)
    past = np.zeros((B, L), dtype=np.float32)
    for i, c in enumerate(contexts):
        c = c[-context_max:]  # most recent
        past[i, :len(c)] = c
        # left-pad with the first value if shorter (model handles via mask
        # implicitly; padding with zeros works since we use mean predictions).
        if len(c) < L:
            past[i, len(c):] = c[-1]
    x = torch.from_numpy(past).to(device)
    # freq=0 is the high-frequency band (TimesFM convention: 0=daily/hourly,
    # 1=weekly/monthly, 2=quarterly/yearly).
    freq = torch.zeros(B, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(past_values=x, freq=freq)
    pred = out.mean_predictions[:, :prediction_length]
    return pred.detach().cpu().numpy()


def evaluate(series: np.ndarray, train_end: int, val_end: int,
             horizons: list[int], lookback: int = 512,
             stride: int = 1, prefer: str = DEFAULT_MODEL) -> dict:
    """Zero-shot TimesFM eval on the test segment of ``series``.

    Mirrors the Chronos baseline: for target index t the model sees
    ``series[t - lookback - h + 1 : t - h + 1]`` and forecasts ``h`` steps; we
    take the final step as the h-day forecast. Returns a dict keyed by horizon.
    """
    model, model_name, device = _load_model(prefer)
    print(f"[timesfm] loaded {model_name} (device={device}) "
          f"ctx={model.config.context_length} horizon={model.config.horizon_length}")

    series = np.asarray(series, dtype=np.float32).ravel()
    test_idx = np.arange(val_end, len(series))[::stride]
    out: dict = {"model": model_name, "mode": "transformers",
                 "device": str(device), "horizons": {}}

    for h in horizons:
        ok = (test_idx >= lookback + h - 1) & (test_idx < len(series))
        idx = test_idx[ok]
        if idx.size == 0:
            continue
        contexts = [series[t - lookback - h + 1: t - h + 1] for t in idx]
        targets = series[idx]
        persistence = series[idx - h]

        t0 = time.time()
        chunk = 32  # keep VRAM modest on the laptop 4090
        preds = np.empty((len(contexts), h), dtype=np.float32)
        for i in range(0, len(contexts), chunk):
            preds[i:i + chunk] = _predict_batch(
                model, contexts[i:i + chunk], prediction_length=h,
            )
        yhat_h = preds[:, -1]
        elapsed = time.time() - t0

        rmse = env_metrics.rmse(targets, yhat_h)
        mae = env_metrics.mae(targets, yhat_h)
        acc = env_metrics.anomaly_correlation(targets, yhat_h)
        skill = env_metrics.skill_score(
            env_metrics.rmse(targets, yhat_h),
            env_metrics.rmse(targets, persistence),
            higher_is_better=False,
        )
        out["horizons"][str(h)] = {
            "rmse": float(rmse), "mae": float(mae), "acc": float(acc),
            "skill_vs_persistence": float(skill),
            "n_eval_targets": int(len(idx)), "wall_time_sec": round(elapsed, 2),
            "lookback": int(lookback),
        }
        print(f"[timesfm]  h={h:>2}d  n={len(idx):>4}  "
              f"RMSE={rmse:.3f}  skill_vs_persistence={skill:+.3f}  ({elapsed:.1f}s)")

    return out


def predict(model, contexts, prediction_length: int) -> np.ndarray:
    """Public helper kept for the unit test (small synthetic shape check)."""
    return _predict_batch(model, list(contexts), prediction_length)
