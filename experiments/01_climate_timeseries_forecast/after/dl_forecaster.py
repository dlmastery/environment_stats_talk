"""AFTER — a small PyTorch LSTM forecaster for daily 2m temperature.

This is the "agentic / AI-for-science" side of the before/after pair. It is
deliberately *small* (a 1-2 layer LSTM head) so it trains in seconds on CPU for
the unit tests, yet it AUTO-USES CUDA when available (the repo machine has an
RTX 4090) for the full headline run.

Design
------
- Windowed supervised dataset: each sample is a length-``lookback`` window of past
  t2m mapped to the value ``h`` steps after the window end. Inputs are standardized
  with statistics computed on the TRAIN split only (no leakage).
- Light feature engineering that helps a small net learn the annual cycle cheaply:
  alongside the standardized temperature we feed sin/cos of day-of-year.
- AdamW optimizer, MSE loss, early stopping on the validation RMSE.
- Deterministic seeding so committed results are reproducible.

Foundation models (optional)
-----------------------------
``zero_shot_foundation_baseline`` documents how to drop in a real time-series
foundation model (Amazon Chronos or Google TimesFM) for a *zero-shot* forecast.
It is fully optional: if the package is not installed it returns ``None`` and the
orchestrator simply skips it. We never require the package or any API key, so the
experiment still "runs anywhere".

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from common import metrics  # noqa: E402

__all__ = [
    "TrainConfig",
    "get_device",
    "set_seed",
    "make_windows",
    "LSTMForecaster",
    "train_forecaster",
    "predict",
    "run_after",
    "zero_shot_foundation_baseline",
]


# --------------------------------------------------------------------------- #
# Config & device
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparameters for the AFTER LSTM. Defaults finish a real run quickly."""
    lookback: int = 30          # days of history per window
    horizon: int = 1            # forecast h steps ahead
    hidden: int = 48
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 40
    patience: int = 6           # early-stopping patience (epochs w/o val improvement)
    seed: int = 0


def get_device() -> torch.device:
    """Auto-select CUDA (RTX 4090) when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Windowed dataset
# --------------------------------------------------------------------------- #
def _doy_features(index: pd.DatetimeIndex) -> np.ndarray:
    """sin/cos of day-of-year — a cheap, leak-free encoding of the annual cycle."""
    doy = index.dayofyear.to_numpy().astype(float)
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)  # (n, 2)


def make_windows(series: pd.Series, lookback: int, horizon: int,
                 mean: float | None = None, std: float | None = None):
    """Build (X, y) windows for one chronological block.

    X has shape (n_samples, lookback, n_features=3): standardized t2m + sin/cos doy.
    y has shape (n_samples,): the standardized t2m ``horizon`` steps after the
    window. Standardization stats (``mean``/``std``) should come from TRAIN only;
    if not given they are computed from this block (used for the train block).

    Returns (X, y, mean, std). When the block is too short, returns empty arrays.
    """
    vals = series.to_numpy().astype(float)
    if mean is None or std is None:
        mean = float(np.mean(vals))
        std = float(np.std(vals) + 1e-8)
    z = (vals - mean) / std
    doy = _doy_features(series.index)

    n = len(vals) - lookback - horizon + 1
    if n <= 0:
        empty_x = np.zeros((0, lookback, 3), dtype=np.float32)
        return empty_x, np.zeros((0,), dtype=np.float32), mean, std

    X = np.empty((n, lookback, 3), dtype=np.float32)
    y = np.empty((n,), dtype=np.float32)
    for i in range(n):
        w = slice(i, i + lookback)
        X[i, :, 0] = z[w]
        X[i, :, 1:] = doy[w]
        y[i] = z[i + lookback + horizon - 1]
    return X, y, mean, std


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    """A compact LSTM -> linear head producing a single standardized prediction."""

    def __init__(self, n_features: int = 3, hidden: int = 48,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):                      # x: (B, lookback, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]                   # last time step's hidden state
        return self.head(last).squeeze(-1)     # (B,)


# --------------------------------------------------------------------------- #
# Train / predict
# --------------------------------------------------------------------------- #
def train_forecaster(train_series: pd.Series, val_series: pd.Series,
                     cfg: TrainConfig, device: torch.device | None = None,
                     verbose: bool = False):
    """Train the LSTM with AdamW + early stopping on validation RMSE.

    Returns (model, stats) where ``stats`` carries the standardization mean/std and
    the best validation RMSE (in standardized units) plus the epoch count run.
    """
    device = device or get_device()
    set_seed(cfg.seed)

    Xtr, ytr, mean, std = make_windows(train_series, cfg.lookback, cfg.horizon)
    # Validation windows reuse TRAIN standardization stats (no leakage) and are
    # built on train-tail + val so the first val targets are predictable.
    val_input = pd.concat([train_series.iloc[-(cfg.lookback + cfg.horizon):], val_series])
    Xva, yva, _, _ = make_windows(val_input, cfg.lookback, cfg.horizon, mean, std)

    model = LSTMForecaster(3, cfg.hidden, cfg.num_layers, cfg.dropout).to(device)

    if len(Xtr) == 0:  # degenerate (extremely short series) — return untrained model
        return model, {"mean": mean, "std": std, "best_val_rmse": float("nan"),
                       "epochs_run": 0, "device": str(device)}

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    has_val = len(Xva) > 0
    if has_val:
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).to(device)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        epochs_run += 1

        # Early stopping on standardized val RMSE (falls back to train loss if no val).
        model.eval()
        with torch.no_grad():
            if has_val:
                pred = model(Xva_t)
                cur = float(torch.sqrt(torch.mean((pred - yva_t) ** 2)).item())
            else:
                xb_all = torch.from_numpy(Xtr).to(device)
                yb_all = torch.from_numpy(ytr).to(device)
                cur = float(torch.sqrt(torch.mean((model(xb_all) - yb_all) ** 2)).item())

        if verbose:
            print(f"epoch {epoch + 1:3d}  val_rmse(z)={cur:.4f}")

        if cur < best_val - 1e-5:
            best_val = cur
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    stats = {"mean": mean, "std": std, "best_val_rmse": best_val,
             "epochs_run": epochs_run, "device": str(device)}
    return model, stats


def predict(model, hist_series: pd.Series, test_series: pd.Series, cfg: TrainConfig,
            stats: dict, device: torch.device | None = None):
    """Forecast the test block ``horizon`` steps ahead.

    ``hist_series`` is the chronological data immediately preceding the test block
    (typically train+val) so every test target has a full lookback window. Uses the
    TRAIN standardization stats stored in ``stats``; de-standardizes predictions
    back to deg C.

    Returns (y_true, y_pred) aligned 1-D arrays over the predictable test targets.
    """
    device = device or get_device()
    mean, std = stats["mean"], stats["std"]

    # Build windows over [tail-of-hist + test] so the first test targets are covered.
    context = pd.concat([hist_series.iloc[-(cfg.lookback + cfg.horizon):], test_series])
    X, _, _, _ = make_windows(context, cfg.lookback, cfg.horizon, mean, std)

    model.eval()
    with torch.no_grad():
        preds_z = model(torch.from_numpy(X).to(device)).cpu().numpy().ravel()
    y_pred = preds_z * std + mean

    # The windows produced exactly len(test_series) targets (the test block).
    y_true = test_series.to_numpy().astype(float).ravel()
    n = min(len(y_true), len(y_pred))
    return y_true[-n:], y_pred[-n:]


def run_after(train_series: pd.Series, val_series: pd.Series, test_series: pd.Series,
              cfg: TrainConfig, verbose: bool = False,
              device: torch.device | None = None) -> dict:
    """Train the AFTER model and evaluate it on the test block.

    ``device`` defaults to the auto-selected device (CUDA when available, e.g. the
    RTX 4090); pass ``torch.device("cpu")`` to force CPU (used by the fast tests).

    Returns a metrics dict:
        {"rmse":.., "mae":.., "acc":.., "epochs_run":.., "device":..,
         "y_true":[...], "y_pred":[...]}
    """
    device = device or get_device()
    model, stats = train_forecaster(train_series, val_series, cfg, device, verbose)
    hist = pd.concat([train_series, val_series])
    y_true, y_pred = predict(model, hist, test_series, cfg, stats, device)
    return {
        "rmse": metrics.rmse(y_true, y_pred),
        "mae": metrics.mae(y_true, y_pred),
        "acc": metrics.anomaly_correlation(y_true, y_pred),
        "epochs_run": stats["epochs_run"],
        "device": stats["device"],
        "best_val_rmse_z": stats["best_val_rmse"],
        "y_true": np.asarray(y_true, float).tolist(),
        "y_pred": np.asarray(y_pred, float).tolist(),
    }


# --------------------------------------------------------------------------- #
# OPTIONAL: zero-shot foundation model baseline (Chronos / TimesFM)
# --------------------------------------------------------------------------- #
def zero_shot_foundation_baseline(train_series: pd.Series, test_series: pd.Series,
                                  h: int = 1, model_name: str = "amazon/chronos-t5-small"):
    """OPTIONAL zero-shot forecast with a real TS foundation model.

    This documents the "AI-for-science" frontier: pretrained time-series foundation
    models (Amazon **Chronos**, Google **TimesFM**) can forecast a *new* series with
    NO task-specific training. We never require these packages or any API key — if
    the dependency is absent this function returns ``None`` and the orchestrator
    skips it, so the experiment still runs anywhere.

    To enable Chronos:  ``pip install chronos-forecasting``  (pulls a small T5
    checkpoint from the Hugging Face Hub on first use; needs network once).
    To enable TimesFM:  ``pip install timesfm``.

    Returns ``(y_true, y_pred)`` aligned 1-D arrays, or ``None`` if unavailable.
    """
    try:
        from chronos import ChronosPipeline  # type: ignore
    except Exception:
        return None  # package not installed -> skip cleanly

    try:
        pipe = ChronosPipeline.from_pretrained(
            model_name, device_map=("cuda" if torch.cuda.is_available() else "cpu"),
            torch_dtype=torch.float32,
        )
        tr = np.asarray(train_series, dtype=float).ravel()
        te = np.asarray(test_series, dtype=float).ravel()
        context = torch.tensor(tr, dtype=torch.float32)
        # Single multi-step forecast of length len(test); take the median quantile.
        forecast = pipe.predict(context, prediction_length=len(te))
        median = np.median(forecast[0].numpy(), axis=0)
        return te, np.asarray(median, dtype=float).ravel()
    except Exception:
        return None  # any runtime/network issue -> skip cleanly


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common import daily_temperature, time_split

    df = daily_temperature(n_years=6, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    cfg = TrainConfig(epochs=8, lookback=30, horizon=1)
    res = run_after(tr["t2m"], va["t2m"], te["t2m"], cfg, verbose=True)
    print(f"AFTER rmse={res['rmse']:.3f} mae={res['mae']:.3f} acc={res['acc']:.3f} "
          f"epochs={res['epochs_run']} device={res['device']}")
