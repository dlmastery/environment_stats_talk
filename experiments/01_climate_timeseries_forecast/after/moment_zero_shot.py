"""Real zero-shot foundation-model baseline for Exp01 using MOMENT.

MOMENT (Goswami et al., 2024, ICML; arXiv:2402.03885) is a family of pretrained
transformer encoders for time-series. The **forecasting head** is *not*
pretrained (the model warns about this), so a fair "zero-shot" path uses the
**pretrained reconstruction head**: we feed a window with the future positions
masked out and the encoder + reconstruction head fill them.

This is the protocol used in the MOMENT paper's "zero-shot imputation" setting
and is the only honest zero-shot use of MOMENT for forecasting.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

try:  # pragma: no cover - environmental shim
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics as env_metrics  # noqa: E402

DEFAULT_MODEL = "AutonLab/MOMENT-1-small"
# MOMENT's pretrained patch length and fixed input length:
MOMENT_INPUT_LEN = 512  # fixed; smaller variants also use 512


def _load_model(prefer: str = DEFAULT_MODEL):
    """Load MOMENT in reconstruction mode (the pretrained head)."""
    from momentfm import MOMENTPipeline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MOMENTPipeline.from_pretrained(
            prefer, model_kwargs={"task_name": "reconstruction"},
        )
        model.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, prefer, device


def _predict_batch(model, contexts: list[np.ndarray], prediction_length: int) -> np.ndarray:
    """Return reconstructed predictions of shape (B, prediction_length).

    Build inputs of length 512: most recent ``512 - prediction_length`` true
    observations + ``prediction_length`` padding positions whose ``input_mask``
    is zero. MOMENT's pretrained reconstruction head fills the masked tail.
    """
    device = next(model.parameters()).device
    B = len(contexts)
    L = MOMENT_INPUT_LEN
    pad = int(prediction_length)
    keep = L - pad
    if keep <= 0:
        raise ValueError(
            f"prediction_length={pad} too large for MOMENT input length {L}"
        )
    x = np.zeros((B, 1, L), dtype=np.float32)
    mask = np.ones((B, L), dtype=np.float32)
    for i, c in enumerate(contexts):
        c = c[-keep:]  # most recent observations
        x[i, 0, :len(c)] = c
        # left-pad with the first value if context shorter than ``keep``
        if len(c) < keep:
            x[i, 0, len(c):keep] = c[0]
        # mask out the future positions we want to predict
        mask[i, -pad:] = 0.0
    xt = torch.from_numpy(x).to(device)
    mt = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        out = model(x_enc=xt, input_mask=mt)
    # ``reconstruction`` has shape (B, 1, L); take the last ``pad`` positions.
    rec = out.reconstruction[:, 0, -pad:]
    return rec.detach().cpu().numpy()


def evaluate(series: np.ndarray, train_end: int, val_end: int,
             horizons: list[int], lookback: int = MOMENT_INPUT_LEN,
             stride: int = 1, prefer: str = DEFAULT_MODEL) -> dict:
    """Zero-shot MOMENT eval (reconstruction-as-forecasting) on the test segment.

    Mirrors the Chronos/TimesFM baselines: for each target index t the model
    sees ``series[t - lookback - h + 1 : t - h + 1]`` (most recent ``lookback - h``
    observations are used as context; the trailing ``h`` positions are masked
    and reconstructed). The final reconstructed step is the h-day forecast.
    """
    model, model_name, device = _load_model(prefer)
    print(f"[moment] loaded {model_name} (device={device}, head=reconstruction)")

    series = np.asarray(series, dtype=np.float32).ravel()
    test_idx = np.arange(val_end, len(series))[::stride]
    out: dict = {"model": model_name, "mode": "reconstruction",
                 "device": str(device), "horizons": {}}

    for h in horizons:
        ok = (test_idx >= lookback + h - 1) & (test_idx < len(series))
        idx = test_idx[ok]
        if idx.size == 0:
            continue
        # Pass only the most-recent ``lookback`` observations; we mask the
        # trailing ``h`` positions internally to perform reconstruction-forecast.
        contexts = [series[t - lookback + 1: t - h + 1] for t in idx]
        targets = series[idx]
        persistence = series[idx - h]

        t0 = time.time()
        chunk = 64
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
        print(f"[moment]   h={h:>2}d  n={len(idx):>4}  "
              f"RMSE={rmse:.3f}  skill_vs_persistence={skill:+.3f}  ({elapsed:.1f}s)")

    return out


def predict(model, contexts, prediction_length: int) -> np.ndarray:
    """Public helper kept for the unit test (small synthetic shape check)."""
    return _predict_batch(model, list(contexts), prediction_length)
