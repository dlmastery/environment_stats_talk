"""Real zero-shot foundation-model baseline for Exp01 using Salesforce Moirai.

Moirai (Woo et al., 2024, ICML; arXiv:2402.02592) is a universal time-series
foundation model that supports zero-shot probabilistic forecasting via a
masked-encoder + flow-matching decoder. We load it directly from HF and call
the raw ``MoiraiForecast.forward`` to keep the eval path identical to the
Chronos / TimesFM / MOMENT wrappers (no GluonTS data adapters).

HF model: Salesforce/moirai-1.0-R-small (or -base).
"""
from __future__ import annotations

import sys
import time
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

DEFAULT_MODEL = "Salesforce/moirai-1.0-R-small"


def _build_forecast_model(prefer: str, context_length: int, prediction_length: int,
                          num_samples: int = 20, patch_size: int = 8):
    """Build a MoiraiForecast at the requested (context, horizon) lengths.

    Moirai requires both ``context_length`` and ``prediction_length`` to be
    multiples of ``patch_size`` (it patches the sequence into fixed-size chunks
    of size 8/16/32/64/128). We round up both to multiples of ``patch_size``
    and let the caller trim outputs back to the requested horizon.
    """
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    # round both lengths up to the nearest multiple of patch_size
    ctx = int(np.ceil(context_length / patch_size) * patch_size)
    pred = int(np.ceil(prediction_length / patch_size) * patch_size)
    module = MoiraiModule.from_pretrained(prefer)
    fcst = MoiraiForecast(
        module=module,
        prediction_length=pred,
        context_length=ctx,
        patch_size=patch_size,
        num_samples=int(num_samples),
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fcst = fcst.to(device).eval()
    return fcst, device, ctx, pred


def _predict_batch(model, contexts: list[np.ndarray], prediction_length: int,
                   num_samples: int = 20) -> np.ndarray:
    """Return median predictions of shape (B, prediction_length)."""
    device = next(model.parameters()).device
    B = len(contexts)
    L = max(len(c) for c in contexts)
    past = np.zeros((B, L, 1), dtype=np.float32)
    observed = np.zeros((B, L, 1), dtype=bool)
    is_pad = np.ones((B, L), dtype=bool)
    for i, c in enumerate(contexts):
        past[i, -len(c):, 0] = c  # right-align observations
        observed[i, -len(c):, 0] = True
        is_pad[i, -len(c):] = False
    past_t = torch.from_numpy(past).to(device)
    observed_t = torch.from_numpy(observed).to(device)
    is_pad_t = torch.from_numpy(is_pad).to(device)
    with torch.no_grad():
        samples = model(
            past_target=past_t,
            past_observed_target=observed_t,
            past_is_pad=is_pad_t,
            num_samples=num_samples,
        )
    # samples: (B, num_samples, prediction_length, target_dim=1)
    if samples.dim() == 4:
        samples = samples.squeeze(-1)
    # median over samples
    med = samples.median(dim=1).values  # (B, prediction_length)
    return med.detach().cpu().numpy()


def evaluate(series: np.ndarray, train_end: int, val_end: int,
             horizons: list[int], lookback: int = 200,
             stride: int = 1, prefer: str = DEFAULT_MODEL,
             num_samples: int = 20) -> dict:
    """Zero-shot Moirai eval on the test segment of ``series``.

    Mirrors the other foundation baselines: for each target index t the model
    sees ``series[t - lookback - h + 1 : t - h + 1]`` and forecasts ``h`` steps;
    we take the median sample and use its final step as the h-day forecast.

    Because Moirai's prediction length is fixed at construction, we build a
    fresh ``MoiraiForecast`` per horizon (cheap; module weights are shared).
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    test_idx = np.arange(val_end, len(series))[::stride]
    out: dict = {"model": prefer, "mode": "moirai", "horizons": {}}

    for h in horizons:
        ok = (test_idx >= lookback + h - 1) & (test_idx < len(series))
        idx = test_idx[ok]
        if idx.size == 0:
            continue
        model, device, ctx_used, pred_used = _build_forecast_model(
            prefer, context_length=lookback, prediction_length=h,
            num_samples=num_samples,
        )
        out.setdefault("device", str(device))
        # use the patch-aligned context length when slicing inputs
        contexts = [series[t - ctx_used - h + 1: t - h + 1] for t in idx]
        targets = series[idx]
        persistence = series[idx - h]

        t0 = time.time()
        chunk = 32
        preds = np.empty((len(contexts), pred_used), dtype=np.float32)
        for i in range(0, len(contexts), chunk):
            preds[i:i + chunk] = _predict_batch(
                model, contexts[i:i + chunk], prediction_length=pred_used,
                num_samples=num_samples,
            )
        # take the requested-horizon step (h-1 index since we want the h-step-ahead
        # value, and we've predicted pred_used >= h steps starting at offset 0)
        yhat_h = preds[:, h - 1]
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
        print(f"[moirai]   h={h:>2}d  n={len(idx):>4}  "
              f"RMSE={rmse:.3f}  skill_vs_persistence={skill:+.3f}  ({elapsed:.1f}s)")

    return out


def predict(model, contexts, prediction_length: int, num_samples: int = 10) -> np.ndarray:
    """Public helper kept for the unit test (small synthetic shape check).

    NOTE: ``model`` must be a pre-built ``MoiraiForecast`` whose
    ``prediction_length`` matches the requested ``prediction_length``.
    """
    return _predict_batch(model, list(contexts), prediction_length, num_samples=num_samples)


def build_for_test(prefer: str = DEFAULT_MODEL, context_length: int = 32,
                   prediction_length: int = 8, num_samples: int = 4):
    """Tiny helper used by the import-and-shape unit test."""
    model, device, ctx, pred = _build_forecast_model(
        prefer, context_length=context_length, prediction_length=prediction_length,
        num_samples=num_samples,
    )
    return model, device, ctx, pred
