"""AFTER -- a gradient-boosted PM2.5 nowcast that USES the weather covariates.

This is the "agentic / AI-for-science" side of the before/after pair. A
``GradientBoostingRegressor`` (scikit-learn, CPU) predicts the next PM2.5 value from
a feature vector that the BEFORE baselines never see:

  * **Weather covariates** -- wind speed (ventilation), temperature, and
    boundary-layer height (the dilution volume / inversion proxy), at the **nowcast
    time** (lag 0) AND short lags, plus a few rolling summaries (recent wind, recent
    boundary-layer). These are the genuine drivers of PM2.5 spikes. Using
    contemporaneous meteorology is the realistic *nowcast* setting: met stations
    (and reanalysis) report wind/temperature/boundary-layer in real time, while the
    reference PM2.5 monitor is sparse/delayed -- so we estimate current PM2.5 from
    current weather. The BEFORE baselines cannot use any of this (history only).
  * **Calendar features** -- cyclical hour-of-day (sin/cos), day-of-week, and a
    weekend flag (the traffic/emission cycle).
  * **A short PM2.5 history** -- a few autoregressive lags, so the model also has
    the self-persistence the BEFORE side relies on. (The AFTER win therefore comes
    from the *added* weather information, not from withholding history.)

Why it wins
-----------
PM2.5 spikes are caused by ventilation collapsing (wind drop + a shallow nocturnal
/ inversion boundary layer) faster than a history-only model can infer from PM2.5
alone. A model that reads the weather sees the cause *as it happens*, so it leads
rather than lags the episodes -- improving both RMSE and the exceedance F1. Remove
the covariates (``use_weather=False``) and the model collapses back to roughly
persistence -- the honest ablation documented in the README.

A scikit-learn ``MLPRegressor`` alternative is provided (``model="mlp"``) for the
"small neural net" variant; both are CPU-friendly and deterministic (fixed
``random_state``). No GPU needed; if you prefer torch, the same feature matrix
plugs into any regressor.

Contract (shared with the BEFORE side):

    run_after(train_df, test_df, ...) -> dict with
        {"rmse":.., "mae":.., "y_true":[...], "y_pred":[...], "model":.., "features":..}

Run from the REPO ROOT so ``import common`` resolves.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXP_DIR = Path(__file__).resolve().parents[1]   # experiments/07_air_quality_nowcast
for _p in (str(_REPO_ROOT), str(_EXP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from aq_metrics import rmse, mae  # type: ignore  # noqa: E402

__all__ = [
    "GBMConfig",
    "build_feature_frame",
    "fit_model",
    "run_after",
]

_TARGET = "pm25"
_WEATHER_COLS = ("wind", "temp", "boundary_layer")


@dataclass
class GBMConfig:
    """Hyperparameters for the AFTER nowcast. Defaults finish in seconds on CPU."""
    model: str = "gbm"                 # "gbm" (GradientBoosting) or "mlp"
    pm_lags: int = 3                   # PM2.5 autoregressive lags (shared with BEFORE idea)
    weather_now: bool = True           # use contemporaneous (lag-0) weather (nowcast)
    weather_lags: int = 2              # additional past lags applied to each covariate
    roll_windows: tuple[int, ...] = (3, 6)  # rolling means of wind & boundary layer
    use_weather: bool = True           # the headline knob; False -> history-only ablation
    n_estimators: int = 300            # GBM trees
    learning_rate: float = 0.05        # GBM shrinkage
    max_depth: int = 3                 # GBM tree depth
    subsample: float = 0.9             # GBM stochastic subsampling
    mlp_hidden: tuple[int, ...] = (64, 32)  # MLP architecture (model="mlp")
    mlp_max_iter: int = 400
    seed: int = 0
    feature_names_: list[str] = field(default_factory=list, repr=False)


def _cyclical(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    ang = 2 * np.pi * values / period
    return np.sin(ang), np.cos(ang)


def build_feature_frame(df: pd.DataFrame, cfg: GBMConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the leak-free feature matrix and one-step-ahead target.

    Each row predicts ``pm25(t)`` from:
      * PM2.5 lags y(t-1)...y(t-pm_lags)  -- the only thing the BEFORE side has too
      * (if use_weather) each weather covariate at the nowcast time t
        (``weather_now``) and at lags 1..weather_lags, plus rolling means of wind &
        boundary_layer over ``roll_windows`` ending at t-1. Contemporaneous weather
        is legitimate for a nowcast (met is observed in real time); it never
        includes the PM2.5 target, so there is no target leakage.
      * cyclical hour + day-of-week and the weekend flag for time t (calendar is
        known in advance, so using time-t calendar is leak-free)

    Returns (X, y, feature_names). Rows lacking full history are dropped.
    """
    n = len(df)
    pm = df[_TARGET].to_numpy().astype(float)
    feats: dict[str, np.ndarray] = {}

    # PM2.5 autoregressive lags (the self-persistence the BEFORE side also uses).
    for k in range(1, cfg.pm_lags + 1):
        col = np.full(n, np.nan)
        col[k:] = pm[:-k]
        feats[f"pm_lag{k}"] = col

    if cfg.use_weather:
        start_lag = 0 if cfg.weather_now else 1
        for name in _WEATHER_COLS:
            v = df[name].to_numpy().astype(float)
            for k in range(start_lag, cfg.weather_lags + 1):
                if k == 0:
                    feats[f"{name}_now"] = v.copy()   # contemporaneous (nowcast)
                    continue
                col = np.full(n, np.nan)
                col[k:] = v[:-k]
                feats[f"{name}_lag{k}"] = col
        # Rolling means of the ventilation drivers, ending at t-1 (shifted by 1).
        for name in ("wind", "boundary_layer"):
            s = pd.Series(df[name].to_numpy().astype(float))
            for w in cfg.roll_windows:
                roll = s.shift(1).rolling(w).mean().to_numpy()
                feats[f"{name}_roll{w}"] = roll

    # Calendar features for time t (known in advance -> leak-free).
    hour = df["hour"].to_numpy().astype(float)
    dow = df["dow"].to_numpy().astype(float)
    h_sin, h_cos = _cyclical(hour, 24.0)
    d_sin, d_cos = _cyclical(dow, 7.0)
    feats["hour_sin"], feats["hour_cos"] = h_sin, h_cos
    feats["dow_sin"], feats["dow_cos"] = d_sin, d_cos
    feats["is_weekend"] = df["is_weekend"].to_numpy().astype(float)

    X = pd.DataFrame(feats, index=df.index)
    y = pd.Series(pm, index=df.index, name=_TARGET)
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    return X, y, list(X.columns)


def fit_model(X: np.ndarray, y: np.ndarray, cfg: GBMConfig):
    """Construct and fit the AFTER regressor (GBM or MLP). Deterministic seed."""
    if cfg.model == "mlp":
        est = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=cfg.mlp_hidden,
                                 max_iter=cfg.mlp_max_iter, random_state=cfg.seed,
                                 early_stopping=True, n_iter_no_change=15)),
        ])
    else:
        est = GradientBoostingRegressor(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth, subsample=cfg.subsample,
            random_state=cfg.seed,
        )
    est.fit(X, y)
    return est


def run_after(train_df: pd.DataFrame, test_df: pd.DataFrame,
              cfg: GBMConfig | None = None) -> dict:
    """Train the AFTER nowcast on train and evaluate one-step-ahead on test.

    Leak-free: the test feature rows are built from a context that prepends a short
    train tail (so the first test targets get full lag/rolling history), but only
    rows mapping to actual test timestamps are scored. Returns
    {"rmse","mae","y_true":[...],"y_pred":[...],"model":..,"features":[...],"use_weather":..}.
    """
    cfg = cfg or GBMConfig()

    Xtr, ytr, names = build_feature_frame(train_df, cfg)
    est = fit_model(Xtr.to_numpy(), ytr.to_numpy(), cfg)

    # Build test features with a train-tail warm-up so lags/rollings are complete.
    warm = max(cfg.pm_lags, cfg.weather_lags, max(cfg.roll_windows, default=1)) + 1
    context = pd.concat([train_df.iloc[-warm:], test_df])
    Xte_full, yte_full, _ = build_feature_frame(context, cfg)
    # Keep only the rows whose timestamp belongs to the actual test block.
    in_test = Xte_full.index.isin(test_df.index)
    Xte, yte = Xte_full[in_test], yte_full[in_test]

    y_pred = np.clip(est.predict(Xte[names].to_numpy()), 0.0, None)
    y_true = yte.to_numpy().astype(float)

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "y_true": np.asarray(y_true, float).tolist(),
        "y_pred": np.asarray(y_pred, float).tolist(),
        "model": cfg.model,
        "features": names,
        "n_features": len(names),
        "use_weather": cfg.use_weather,
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from common.synthetic_airquality import synthetic_pm25
    from common.synthetic_climate import time_split

    df = synthetic_pm25(n_days=90, seed=0, freq="h")
    tr, va, te = time_split(df, 0.7, 0.15)
    for use_w in (True, False):
        res = run_after(pd.concat([tr, va]), te, GBMConfig(use_weather=use_w))
        tag = "with weather" if use_w else "history-only (ablation)"
        print(f"AFTER GBM ({tag:24s}) rmse={res['rmse']:.3f} mae={res['mae']:.3f} "
              f"n_features={res['n_features']}")
