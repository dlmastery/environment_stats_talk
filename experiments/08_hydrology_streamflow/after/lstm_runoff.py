"""AFTER — an LSTM rainfall-runoff model (the well-known LSTM-streamflow result).

This is the "agentic / AI-for-science" side of the before/after pair. An LSTM reads
a lookback window of daily meteorological forcing — ``[precip, temperature, pet]``
(plus a cheap day-of-year encoding) — and predicts the day's streamflow. Because the
LSTM carries an internal cell state across the window, it learns the **nonlinear,
state-dependent catchment memory** (antecedent soil moisture, snow accumulation /
melt, routing lag) that the linear BEFORE baseline structurally cannot represent.

Design
------
- Windowed supervised dataset: each sample is a length-``lookback`` window of the
  three forcings + sin/cos day-of-year, mapped to streamflow at the window's last
  day. Inputs are standardized with TRAIN-only statistics (no leakage); the target
  is standardized too and de-standardized at prediction time.
- Compact LSTM (1-2 layers) -> linear head producing a single value. Predictions
  are clipped at zero (negative streamflow is unphysical).
- AdamW optimizer, MSE loss, early stopping on validation NSE (the hydrology skill
  metric), deterministic seeding.
- **CPU for the fast tests; auto-CUDA (RTX 4090) for the headline run** via
  ``get_device()``.

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1]   # experiments/08_hydrology_streamflow
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from hydro_metrics import nse, kge, rmse  # type: ignore  # noqa: E402

__all__ = [
    "LSTMConfig",
    "get_device",
    "set_seed",
    "FORCING_COLS",
    "make_windows",
    "LSTMRunoff",
    "train_runoff",
    "predict",
    "run_after",
]

FORCING_COLS = ("precip", "temperature", "pet")
_N_FEATURES = len(FORCING_COLS) + 2   # + sin/cos day-of-year


@dataclass
class LSTMConfig:
    """Hyperparameters for the AFTER LSTM. Defaults finish a modest run in minutes."""
    lookback: int = 45          # days of forcing history per window (catchment memory)
    hidden: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 60
    patience: int = 12          # early-stopping patience on validation MSE
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


def make_windows(df: pd.DataFrame, lookback: int,
                 feat_mean: np.ndarray | None = None,
                 feat_std: np.ndarray | None = None,
                 y_mean: float | None = None, y_std: float | None = None):
    """Build (X, y) windows from a chronological catchment block.

    X has shape (n_samples, lookback, n_features): standardized forcings
    ``[precip, temperature, pet]`` + sin/cos day-of-year. y is the standardized
    streamflow at each window's LAST day. Standardization stats should come from
    TRAIN only (pass them in for val/test); if omitted they are computed here.

    Returns (X, y, feat_mean, feat_std, y_mean, y_std). Empty arrays if too short.
    """
    forcings = df[list(FORCING_COLS)].to_numpy().astype(float)   # (n, 3)
    flow = df["streamflow"].to_numpy().astype(float)             # (n,)
    doy = _doy_features(df.index)                                # (n, 2)

    if feat_mean is None or feat_std is None:
        feat_mean = forcings.mean(axis=0)
        feat_std = forcings.std(axis=0) + 1e-8
    if y_mean is None or y_std is None:
        y_mean = float(flow.mean())
        y_std = float(flow.std() + 1e-8)

    fz = (forcings - feat_mean) / feat_std
    feats = np.concatenate([fz, doy], axis=1)                    # (n, 5)
    yz = (flow - y_mean) / y_std

    n = len(df) - lookback + 1
    if n <= 0:
        empty_x = np.zeros((0, lookback, _N_FEATURES), dtype=np.float32)
        return empty_x, np.zeros((0,), dtype=np.float32), feat_mean, feat_std, y_mean, y_std

    X = np.empty((n, lookback, _N_FEATURES), dtype=np.float32)
    y = np.empty((n,), dtype=np.float32)
    for i in range(n):
        X[i] = feats[i:i + lookback]
        y[i] = yz[i + lookback - 1]    # predict the window's last day
    return X, y, feat_mean, feat_std, y_mean, y_std


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMRunoff(nn.Module):
    """A compact LSTM -> linear head producing one standardized streamflow value."""

    def __init__(self, n_features: int = _N_FEATURES, hidden: int = 64,
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
def train_runoff(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: LSTMConfig,
                 device: torch.device | None = None, verbose: bool = False):
    """Train the LSTM with AdamW + early stopping on validation NSE.

    Returns (model, stats) where ``stats`` carries the standardization statistics,
    the best validation NSE, epochs run, and the device string.
    """
    device = device or get_device()
    set_seed(cfg.seed)

    Xtr, ytr, fmean, fstd, ymean, ystd = make_windows(train_df, cfg.lookback)
    # Val windows reuse TRAIN stats (no leakage) and prepend a train tail so the
    # first val targets get a full lookback window.
    val_input = pd.concat([train_df.iloc[-(cfg.lookback - 1):], val_df]) \
        if cfg.lookback > 1 else val_df
    Xva, yva, *_ = make_windows(val_input, cfg.lookback, fmean, fstd, ymean, ystd)

    model = LSTMRunoff(_N_FEATURES, cfg.hidden, cfg.num_layers, cfg.dropout).to(device)

    if len(Xtr) == 0:  # degenerate (extremely short series)
        return model, {"feat_mean": fmean, "feat_std": fstd, "y_mean": ymean,
                       "y_std": ystd, "best_val_nse": float("nan"),
                       "epochs_run": 0, "device": str(device)}

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    has_val = len(Xva) > 0
    if has_val:
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).to(device)
        yva_real = yva * ystd + ymean   # de-standardized val targets for NSE report
    else:
        Xtr_t = torch.from_numpy(Xtr).to(device)
        ytr_t = torch.from_numpy(ytr).to(device)

    # Early stop on validation MSE (in standardized units): a smooth, monotone-ish
    # criterion that is far more stable than NSE for stopping. We still record the
    # NSE achieved at the best-MSE epoch for reporting.
    best_mse = float("inf")
    best_nse = float("nan")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        epochs_run += 1

        model.eval()
        with torch.no_grad():
            if has_val:
                pred_z = model(Xva_t)
                cur_mse = float(torch.mean((pred_z - yva_t) ** 2).item())
                cur_nse = nse(yva_real,
                              np.clip(pred_z.cpu().numpy().ravel() * ystd + ymean, 0.0, None))
            else:
                pred_z = model(Xtr_t)
                cur_mse = float(torch.mean((pred_z - ytr_t) ** 2).item())
                cur_nse = nse(ytr * ystd + ymean,
                              np.clip(pred_z.cpu().numpy().ravel() * ystd + ymean, 0.0, None))

        if verbose:
            print(f"epoch {epoch + 1:3d}  val_mse(z)={cur_mse:.4f}  val_nse={cur_nse:.4f}")

        if np.isfinite(cur_mse) and cur_mse < best_mse - 1e-6:
            best_mse = cur_mse
            best_nse = cur_nse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    stats = {"feat_mean": fmean, "feat_std": fstd, "y_mean": ymean, "y_std": ystd,
             "best_val_nse": float(best_nse), "epochs_run": epochs_run,
             "device": str(device)}
    return model, stats


def predict(model, hist_df: pd.DataFrame, test_df: pd.DataFrame, cfg: LSTMConfig,
            stats: dict, device: torch.device | None = None):
    """Forecast streamflow over the test block.

    ``hist_df`` is the chronological data immediately preceding the test block
    (train+val) so every test target has a full lookback window. Uses the TRAIN
    standardization stats in ``stats``; de-standardizes and clips predictions at 0.

    Returns (y_true, y_pred) aligned 1-D arrays over the test targets.
    """
    device = device or get_device()
    fmean, fstd = stats["feat_mean"], stats["feat_std"]
    ymean, ystd = stats["y_mean"], stats["y_std"]

    context = pd.concat([hist_df.iloc[-(cfg.lookback - 1):], test_df]) \
        if cfg.lookback > 1 else test_df
    X, _, *_ = make_windows(context, cfg.lookback, fmean, fstd, ymean, ystd)

    model.eval()
    with torch.no_grad():
        preds_z = model(torch.from_numpy(X).to(device)).cpu().numpy().ravel()
    y_pred = np.clip(preds_z * ystd + ymean, 0.0, None)

    y_true = test_df["streamflow"].to_numpy().astype(float).ravel()
    n = min(len(y_true), len(y_pred))
    return y_true[-n:], y_pred[-n:]


def run_after(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
              cfg: LSTMConfig, verbose: bool = False,
              device: torch.device | None = None) -> dict:
    """Train the AFTER LSTM and evaluate it on the test block.

    ``device`` defaults to the auto-selected device (CUDA when available); pass
    ``torch.device('cpu')`` to force CPU (used by the fast tests).

    Returns: {"rmse","nse","kge","epochs_run","device","best_val_nse",
              "y_true":[...],"y_pred":[...]}.
    """
    device = device or get_device()
    model, stats = train_runoff(train_df, val_df, cfg, device, verbose)
    hist = pd.concat([train_df, val_df])
    y_true, y_pred = predict(model, hist, test_df, cfg, stats, device)
    return {
        "rmse": rmse(y_true, y_pred),
        "nse": nse(y_true, y_pred),
        "kge": kge(y_true, y_pred),
        "epochs_run": stats["epochs_run"],
        "device": stats["device"],
        "best_val_nse": stats["best_val_nse"],
        "y_true": np.asarray(y_true, float).tolist(),
        "y_pred": np.asarray(y_pred, float).tolist(),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_hydrology import synthetic_catchment
    from common.synthetic_climate import time_split

    df = synthetic_catchment(n_years=10, seed=0)
    tr, va, te = time_split(df, 0.7, 0.15)
    cfg = LSTMConfig(epochs=30, lookback=60)
    res = run_after(tr, va, te, cfg, verbose=True)
    print(f"AFTER rmse={res['rmse']:.3f} nse={res['nse']:.3f} kge={res['kge']:.3f} "
          f"epochs={res['epochs_run']} device={res['device']}")
