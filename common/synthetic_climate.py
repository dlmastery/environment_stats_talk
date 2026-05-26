"""Synthetic climate data generators.

These produce *physically plausible* but fully synthetic series so that every
experiment is reproducible offline (deterministic seeds, no network, no CDS API
key). Each generator documents how to swap in the real dataset (ERA5 via cdsapi).

Design choices that make the synthetic data a fair stand-in:
- Temperature: annual + semi-annual seasonal cycle, a configurable warming trend,
  AR(1) "weather" persistence, and occasional heat extremes — so trend tests,
  forecasting skill, and extreme-value analysis all have real signal to find.
- Precipitation: seasonal wet/dry cycle, intermittent (many dry days), and a
  heavy upper tail that intensifies over time — the setting for GEV/POT analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "daily_temperature", "daily_precipitation", "gridded_temperature_field",
    "time_split", "REAL_DATA_NOTE",
]

REAL_DATA_NOTE = (
    "To use real ERA5 instead: `pip install cdsapi xarray`, register at the "
    "Copernicus Climate Data Store, then download 2m_temperature / total_precipitation "
    "and load with xarray. Keep the same DataFrame schema (a 'date' index plus the "
    "variable column) and every downstream script works unchanged."
)


def daily_temperature(
    n_years: int = 20,
    start: str = "2005-01-01",
    seed: int = 0,
    mean_c: float = 18.0,
    annual_amp_c: float = 9.0,
    warming_c_per_decade: float = 0.35,
    ar1: float = 0.78,
    noise_sd_c: float = 1.6,
    heat_extreme_rate: float = 0.012,
    hemisphere: str = "north",
) -> pd.DataFrame:
    """Daily 2m-temperature-like series.

    Returns a DataFrame indexed by date with a single column ``t2m`` (deg C).
    Mimics a mid-latitude station: seasonal cycle + multidecadal warming + AR(1)
    weather noise + sporadic heat spikes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_years * 365, freq="D")
    t = np.arange(len(dates))
    doy = dates.dayofyear.to_numpy()

    phase = 0.0 if hemisphere == "north" else np.pi
    seasonal = annual_amp_c * np.cos(2 * np.pi * (doy - 196) / 365.25 + phase)
    seasonal += 1.5 * np.cos(4 * np.pi * (doy - 30) / 365.25)  # semi-annual
    trend = warming_c_per_decade * (t / 3652.5)

    # AR(1) weather noise
    eps = rng.normal(0.0, noise_sd_c, size=len(dates))
    weather = np.zeros(len(dates))
    for i in range(1, len(dates)):
        weather[i] = ar1 * weather[i - 1] + eps[i]

    # Sporadic heat extremes (more likely in warm season, intensifying with trend)
    extreme = np.zeros(len(dates))
    warm_season = (seasonal > 0).astype(float)
    p_ex = heat_extreme_rate * (0.4 + warm_season) * (1 + trend / 2.0)
    hits = rng.random(len(dates)) < p_ex
    extreme[hits] = rng.gamma(shape=2.5, scale=2.2, size=hits.sum())

    t2m = mean_c + seasonal + trend + weather + extreme
    return pd.DataFrame({"t2m": t2m}, index=dates).rename_axis("date")


def daily_precipitation(
    n_years: int = 20,
    start: str = "2005-01-01",
    seed: int = 1,
    wet_day_prob: float = 0.32,
    seasonal_wet_amp: float = 0.18,
    mean_wet_mm: float = 6.0,
    tail_shape: float = 0.18,
    intensification_per_decade: float = 0.06,
) -> pd.DataFrame:
    """Daily precipitation-like series (mm).

    Intermittent (many zeros), seasonally modulated wet-day probability, and a
    heavy upper tail (generalized-Pareto-style) whose scale intensifies over time
    — the canonical setting for extreme-value analysis (Rx5day, R95p, GEV/POT).
    Returns a DataFrame indexed by date with column ``precip``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_years * 365, freq="D")
    doy = dates.dayofyear.to_numpy()
    tdec = np.arange(len(dates)) / 3652.5

    # Seasonal wet-day probability (wet season mid-year)
    p_wet = wet_day_prob * (1 + seasonal_wet_amp * np.cos(2 * np.pi * (doy - 200) / 365.25))
    is_wet = rng.random(len(dates)) < p_wet

    # Wet-day amounts: gamma body + occasional heavy GPD-like tail
    scale = mean_wet_mm * (1 + intensification_per_decade * tdec)
    body = rng.gamma(shape=1.3, scale=scale / 1.3)
    # Heavy tail multiplier on ~4% of wet days
    heavy = rng.random(len(dates)) < 0.04
    tail = np.where(heavy, (1 - rng.random(len(dates))) ** (-tail_shape), 1.0)
    precip = np.where(is_wet, body * tail, 0.0)
    precip = np.clip(precip, 0.0, None)
    return pd.DataFrame({"precip": precip}, index=dates).rename_axis("date")


@dataclass
class GriddedField:
    """A gridded temperature snapshot with coordinates (for lat-weighted metrics)."""
    field: np.ndarray   # shape (n_lat, n_lon), deg C
    lats: np.ndarray    # degrees, length n_lat
    lons: np.ndarray    # degrees, length n_lon


def gridded_temperature_field(n_lat: int = 32, n_lon: int = 64, seed: int = 2) -> GriddedField:
    """A smooth synthetic temperature field on a lat/lon grid.

    Latitude gradient (warm equator, cold poles) + smooth large-scale anomalies.
    Useful for demonstrating latitude-weighted RMSE/ACC on gridded forecasts.
    """
    rng = np.random.default_rng(seed)
    lats = np.linspace(-87.5, 87.5, n_lat)
    lons = np.linspace(-177.5, 177.5, n_lon)
    lat_grid = lats[:, None]
    base = 30.0 - 0.55 * np.abs(lat_grid)            # equator-to-pole gradient
    # Smooth anomalies via low-frequency sinusoids with random phases
    anom = np.zeros((n_lat, n_lon))
    for k in range(1, 4):
        ph = rng.uniform(0, 2 * np.pi)
        anom += (3.0 / k) * np.sin(k * np.deg2rad(lons)[None, :] + ph) \
            * np.cos(k * np.deg2rad(lats)[:, None])
    field = base + anom + rng.normal(0, 0.4, size=(n_lat, n_lon))
    return GriddedField(field=field, lats=lats, lons=lons)


def time_split(df: pd.DataFrame, train: float = 0.7, val: float = 0.15):
    """Chronological train/val/test split (no shuffling — never leak the future).

    Returns (train_df, val_df, test_df). val/test are the most recent windows.
    """
    n = len(df)
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return df.iloc[:i_tr].copy(), df.iloc[i_tr:i_va].copy(), df.iloc[i_va:].copy()
