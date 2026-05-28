"""Real zero-shot foundation-model baseline for Exp01 using Chronos.

Loads Amazon Chronos (T5- or Bolt-style) from Hugging Face and evaluates it
zero-shot on the **same** synthetic temperature series and **same** chronological
split as the other Exp01 baselines (persistence / seasonal-naive / SARIMA / LSTM),
so the comparison is apples-to-apples. Uses CUDA when available.

This is the genuine "after / foundation model" path — no synthetic stand-in,
no scaffolding. It runs zero-shot: no fine-tuning, just `predict(context, h)`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# On Windows, the default Python SSL stack can fail to verify huggingface.co's
# certificate chain because corporate/AV-rewritten roots live in the Windows
# cert store, not in certifi. ``truststore`` makes Python use the OS truststore
# so the HF Hub download succeeds. Best-effort: silently skipped if absent.
try:  # pragma: no cover - environmental shim
    import truststore as _truststore
    _truststore.inject_into_ssl()
except Exception:
    pass

# Make `import common` resolve when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common import metrics as env_metrics  # noqa: E402


def _load_pipeline(prefer: str = "bolt"):
    """Load a Chronos pipeline; try Bolt first (faster), fall back to T5.

    Returns (pipeline, name, mode) where mode is 'bolt' or 't5'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if prefer == "bolt":
        try:
            from chronos import ChronosBoltPipeline
            name = "amazon/chronos-bolt-small"
            pipe = ChronosBoltPipeline.from_pretrained(name, device_map=device)
            return pipe, name, "bolt"
        except Exception as e:
            print(f"[chronos] Bolt unavailable ({e!r}); falling back to T5.")
    from chronos import ChronosPipeline
    name = "amazon/chronos-t5-small"
    pipe = ChronosPipeline.from_pretrained(name, device_map=device, torch_dtype=dtype)
    return pipe, name, "t5"


def _predict_batch(pipeline, mode: str, contexts: list[torch.Tensor],
                   prediction_length: int, num_samples: int = 20) -> np.ndarray:
    """Return median predictions of shape (B, prediction_length)."""
    if mode == "bolt":
        # ChronosBolt: quantile-based; take the median (q=0.5).
        quantiles, _mean = pipeline.predict_quantiles(
            contexts, prediction_length, quantile_levels=[0.5],
        )
        return quantiles[..., 0].detach().cpu().numpy()
    # ChronosPipeline: sample-based; take the median across samples.
    samples = pipeline.predict(contexts, prediction_length, num_samples=num_samples)
    # samples: (B, num_samples, prediction_length)
    return samples.median(dim=1).values.detach().cpu().numpy()


def evaluate(series: np.ndarray, train_end: int, val_end: int,
             horizons: list[int], lookback: int = 96,
             stride: int = 1, prefer: str = "bolt") -> dict:
    """Zero-shot Chronos eval on the test segment of ``series``.

    Mirrors the other Exp01 baselines:
      - test targets are ``series[val_end:]``,
      - for target index t, the model sees ``series[t - lookback - h + 1 : t - h + 1]``
        (length=lookback, all causal — never leaks the target),
      - we forecast ``h`` steps ahead and take the final step as the h-day forecast.
    Returns a dict keyed by horizon with rmse/mae/acc/skill_vs_persistence + wall time.
    """
    pipe, model_name, mode = _load_pipeline(prefer)
    print(f"[chronos] loaded {model_name} (mode={mode}, device={pipe.model.device})")

    series = np.asarray(series, dtype=np.float32).ravel()
    test_idx = np.arange(val_end, len(series))[::stride]
    out: dict = {"model": model_name, "mode": mode, "horizons": {}}

    for h in horizons:
        # Only positions where we have a full lookback window AND the target exists.
        ok = (test_idx >= lookback + h - 1) & (test_idx < len(series))
        idx = test_idx[ok]
        if idx.size == 0:
            continue
        # Build contexts (one per target) and batch them through the pipeline.
        contexts = [torch.from_numpy(series[t - lookback - h + 1: t - h + 1]).float()
                    for t in idx]
        targets = series[idx]
        persistence = series[idx - h]    # baseline used for skill score

        t0 = time.time()
        # Process in chunks to keep VRAM modest on the laptop 4090.
        chunk = 64
        preds = np.empty((len(contexts), h), dtype=np.float32)
        for i in range(0, len(contexts), chunk):
            preds[i:i + chunk] = _predict_batch(
                pipe, mode, contexts[i:i + chunk], prediction_length=h,
            )
        yhat_h = preds[:, -1]          # the h-step-ahead point forecast
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
        print(f"[chronos]  h={h:>2}d  n={len(idx):>4}  "
              f"RMSE={rmse:.3f}  skill_vs_persistence={skill:+.3f}  ({elapsed:.1f}s)")

    return out
