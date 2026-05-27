"""Synthetic air-quality (PM2.5) data generator driven by synthetic meteorology.

This produces a *physically plausible* but fully synthetic hourly (or daily) PM2.5
record so the air-quality nowcast/forecast experiment is reproducible offline
(deterministic seeds, no network, no OpenAQ API key). It returns a DataFrame whose
PM2.5 column is genuinely driven by weather covariates, so a model that *sees the
weather* can beat a model that only sees PM2.5 history.

Why this generator exists
-------------------------
Surface PM2.5 is, to first order, **emissions / ventilation**. The same emissions
produce clean air on a windy, well-mixed afternoon and a choking haze on a calm
night under a temperature inversion. The drivers we encode are the standard ones in
air-quality science:

1. **Emissions cycle** — a diurnal traffic/combustion pattern (rush-hour peaks) and
   a weekly cycle (weekday > weekend), plus a slow seasonal heating term (more
   combustion in cold months). This is the *source* term.
2. **Ventilation by wind** — wind disperses pollution. We model a synthetic wind
   speed; PM2.5 is reduced by a factor that falls as wind rises (clean-out).
3. **Boundary-layer height / temperature inversions** — at night and in cold,
   stagnant conditions the planetary boundary layer (PBL) collapses, trapping
   emissions in a shallow layer so concentrations spike. We derive a synthetic
   boundary-layer height from temperature, hour, and wind, and PM2.5 scales
   *inversely* with it (shallow layer -> high concentration).
4. **Episodic stagnation events** — multi-day high-pressure stagnation (low wind,
   strong inversion) produces the pollution *episodes* (exceedance spikes) that an
   exceedance-detection model has to catch.

The result is a series with strong, genuinely-predictive weather covariates plus a
self-persistence (yesterday looks like today) that a pure-history baseline can ride
partway — exactly the BEFORE/AFTER tension this experiment studies.

Schema
------
``synthetic_pm25`` returns a DataFrame indexed by ``date`` (a DatetimeIndex at the
chosen frequency) with columns::

    pm25            target, micrograms / m^3, strictly non-negative
    wind            10 m wind speed proxy (m/s)            -- ventilation
    temp            2 m temperature (deg C)                -- inversion driver
    boundary_layer  planetary boundary-layer height (m)    -- dilution volume
    hour            hour of day 0..23 (0 for daily freq)   -- calendar feature
    dow             day of week 0..6 (Mon=0)               -- calendar feature
    is_weekend      1.0 on Sat/Sun else 0.0                -- calendar feature

Swapping in real data
---------------------
``REAL_DATA_NOTE`` documents how to drop in real **OpenAQ** PM2.5 with reanalysis
meteorology. Keep the same schema and every downstream script works unchanged.

Implementation note: the meteorological forcing reuses the repo's climate
generators (``common.synthetic_climate``) for the temperature backbone so the
weather looks like the rest of the repo, then layers on a diurnal cycle and a
synthetic wind field. The submodule is imported directly (it is intentionally NOT
exported from ``common/__init__``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the calibrated daily-temperature generator from the climate module for the
# slow (seasonal + multi-day weather) temperature backbone. Import the submodule
# directly, not the common package __init__.
from common.synthetic_climate import daily_temperature

__all__ = [
    "AirQualityParams",
    "synthetic_pm25",
    "exceedance_threshold",
    "REAL_DATA_NOTE",
]

REAL_DATA_NOTE = (
    "To use real data instead of this synthetic series, pull PM2.5 from OpenAQ "
    "(https://openaq.org; the v3 REST API or the open S3 archive) for one station, "
    "and the matching meteorology (10 m wind speed, 2 m temperature, planetary "
    "boundary-layer height) from a reanalysis such as ERA5 (Copernicus CDS via "
    "cdsapi/xarray) or a local met station. Resample everything to a common "
    "hourly/daily index and assemble a DataFrame with the SAME columns "
    "['pm25', 'wind', 'temp', 'boundary_layer', 'hour', 'dow', 'is_weekend'] "
    "indexed by a 'date' DatetimeIndex. Every script here (baseline, GBM nowcast, "
    "runner, tests) then works unchanged. OpenAQ is one of the verified open "
    "air-quality datasets; ERA5 boundary-layer height is the standard ventilation "
    "proxy."
)


@dataclass
class AirQualityParams:
    """Generator parameters. Defaults give an urban-station-like PM2.5 series with
    clear, genuinely-predictive weather control (wind clears, inversions trap) plus
    episodic stagnation spikes — so a covariate model beats a history-only model."""
    base_emission: float = 14.0       # baseline emission level (sets typical PM2.5)
    diurnal_amp: float = 0.55         # strength of the twice-daily traffic cycle
    weekend_factor: float = 0.78      # weekday->weekend emission multiplier (<1)
    heating_amp: float = 0.45         # extra cold-season combustion (fraction)
    wind_clean: float = 0.42          # how strongly wind ventilates (per m/s, scaled)
    wind_mean: float = 3.2            # mean wind speed (m/s)
    wind_sd: float = 1.1              # wind AR(1) innovation sd
    wind_ar1: float = 0.86            # wind persistence (multi-hour/day weather)
    pbl_min: float = 150.0            # minimum (nocturnal/inversion) PBL height (m)
    pbl_max: float = 1500.0           # maximum (well-mixed afternoon) PBL height (m)
    pbl_ref: float = 600.0            # reference PBL for the dilution scaling (m)
    inversion_strength: float = 1.9   # how hard a shallow PBL traps pollution
    stagnation_rate: float = 0.010    # per-step probability of starting an episode
    stagnation_len: int = 36          # mean episode length (steps; hours if hourly)
    stagnation_boost: float = 2.6     # peak multiplicative boost during an episode
    obs_noise_cv: float = 0.07        # multiplicative observation noise (coeff. var.)
    # Residual AR(1) "stickiness" of PM2.5 itself. Kept modest so the SPIKES are
    # driven by the *weather* (wind drop + collapsing boundary layer), not by raw
    # self-persistence -- this is what lets a covariate model genuinely beat a
    # history-only baseline (and what the history-only ablation exposes).
    persistence_ar1: float = 0.15


def exceedance_threshold(pm25: np.ndarray, quantile: float = 0.90) -> float:
    """A data-driven exceedance threshold (the ``quantile``-th percentile of PM2.5).

    Used to define "spike / unhealthy" events for the spike-detection F1 metric.
    A quantile keeps the positive-class rate sane across configs (vs a fixed WHO/EPA
    cutoff that could be almost-never or almost-always exceeded on synthetic data).
    """
    return float(np.quantile(np.asarray(pm25, dtype=float), quantile))


def _wind_series(n: int, rng: np.random.Generator, p: AirQualityParams,
                 diurnal: np.ndarray) -> np.ndarray:
    """An AR(1) positive wind-speed series with a mild diurnal component.

    Wind tends to be a touch stronger by day (daytime mixing) and persists over
    multiple steps (weather systems). Strictly non-negative.
    """
    eps = rng.normal(0.0, p.wind_sd, size=n)
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = p.wind_ar1 * z[i - 1] + eps[i]
    # Mild daytime enhancement: +20% around midday via the same diurnal phase.
    day_boost = 1.0 + 0.20 * diurnal
    wind = (p.wind_mean + z) * day_boost
    return np.clip(wind, 0.05, None)


def _boundary_layer(temperature: np.ndarray, hour: np.ndarray, wind: np.ndarray,
                    p: AirQualityParams) -> np.ndarray:
    """Synthetic planetary boundary-layer (mixing) height in metres.

    Deep, well-mixed layer on warm, windy afternoons; shallow, stable layer on cold,
    calm nights (a temperature inversion). Driven by a diurnal solar-heating term, a
    temperature anomaly term, and a wind/mechanical-mixing term. Bounded to
    [pbl_min, pbl_max].
    """
    # Diurnal solar term: peaks mid-afternoon (~15:00), minimum pre-dawn.
    solar = np.clip(np.sin(2 * np.pi * (hour - 6.0) / 24.0), 0.0, None)
    # Warm anomalies deepen the layer; cold anomalies (inversions) shrink it.
    temp_anom = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-8)
    mech = wind / (p.wind_mean + 1e-8)              # mechanical mixing from wind
    drive = 0.62 * solar + 0.22 * np.clip(temp_anom, -3, 3) / 3.0 + 0.16 * np.clip(mech, 0, 3) / 3.0
    drive = np.clip(drive, 0.0, 1.0)
    pbl = p.pbl_min + (p.pbl_max - p.pbl_min) * drive
    return np.clip(pbl, p.pbl_min, p.pbl_max)


def _stagnation_episodes(n: int, rng: np.random.Generator,
                         p: AirQualityParams) -> np.ndarray:
    """A non-negative episode-intensity series in [0, ~1].

    Episodes start with probability ``stagnation_rate`` per step and ramp up then
    decay (a triangular envelope) over ~``stagnation_len`` steps — multi-day
    high-pressure stagnation that produces the pollution spikes.
    """
    intensity = np.zeros(n)
    i = 0
    while i < n:
        if rng.random() < p.stagnation_rate:
            length = max(4, int(rng.poisson(p.stagnation_len)))
            length = min(length, n - i)
            peak = length // 2 if length > 1 else 0
            for j in range(length):
                # Triangular rise/fall, scaled to [0,1].
                env = (j + 1) / (peak + 1) if j <= peak else (length - j) / (length - peak + 1e-9)
                intensity[i + j] = max(intensity[i + j], np.clip(env, 0.0, 1.0))
            i += length
        else:
            i += 1
    return intensity


def synthetic_pm25(
    n_days: int = 120,
    start: str = "2021-01-01",
    seed: int = 0,
    freq: str = "h",
    params: AirQualityParams | None = None,
) -> pd.DataFrame:
    """Generate a synthetic PM2.5 record driven by synthetic meteorology.

    Parameters
    ----------
    n_days : number of days to simulate.
    start  : start timestamp of the index.
    seed   : master seed; the temperature backbone uses a derived sub-seed so it is
             decorrelated from wind/episodes but fully deterministic.
    freq   : ``"h"`` for hourly (24 steps/day, the default — gives the diurnal
             traffic + inversion signal) or ``"D"`` for daily aggregates.
    params : :class:`AirQualityParams` (defaults give a clear covariate-driven win).

    Returns
    -------
    DataFrame indexed by ``date`` with columns
    ``['pm25', 'wind', 'temp', 'boundary_layer', 'hour', 'dow', 'is_weekend']``.
    PM2.5 is strictly non-negative.

    The PM2.5 model (per step ``t``)::

        emission_t = base * diurnal(hour) * weekend(dow) * heating(season)
                     * (1 + stagnation_boost * episode_t)
        ventilation_t = 1 / (1 + wind_clean * wind_t)          # wind clears
        dilution_t    = (pbl_ref / pbl_t) ** inversion_strength # inversion traps
        pm_raw_t      = emission_t * ventilation_t * dilution_t
        pm_t          = AR(1) smoothing of pm_raw_t, * obs noise, clipped >= 0
    """
    p = params or AirQualityParams()
    rng = np.random.default_rng(seed)

    if freq.lower() in ("h", "hourly", "1h"):
        freq_code = "h"
        steps_per_day = 24
    elif freq.lower() in ("d", "daily", "1d"):
        freq_code = "D"
        steps_per_day = 1
    else:
        raise ValueError(f"freq must be 'h' or 'D', got {freq!r}")

    n = n_days * steps_per_day
    index = pd.date_range(start=start, periods=n, freq=freq_code)
    hour = index.hour.to_numpy().astype(float)
    dow = index.dayofweek.to_numpy().astype(float)
    doy = index.dayofyear.to_numpy().astype(float)
    is_weekend = (dow >= 5).astype(float)

    # --- Temperature backbone: a daily seasonal+weather series from the climate ---
    # generator, then (for hourly data) add a diurnal swing so nights are cooler.
    n_years = max(1, int(np.ceil(n_days / 365)) + 1)
    temp_daily = daily_temperature(
        n_years=n_years, start=start, seed=seed + 313,
        mean_c=12.0, annual_amp_c=11.0, noise_sd_c=2.0,
        warming_c_per_decade=0.0,
    )["t2m"].to_numpy()
    # Map each step to its day index, then add a diurnal temperature swing.
    day_idx = (np.arange(n) // steps_per_day)
    day_idx = np.clip(day_idx, 0, len(temp_daily) - 1)
    temperature = temp_daily[day_idx]
    if steps_per_day > 1:
        # Coldest pre-dawn (~05:00), warmest mid-afternoon (~15:00).
        diurnal_temp = 5.5 * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
        temperature = temperature + diurnal_temp

    # --- Diurnal emission shape: two rush-hour peaks (~08:00 and ~19:00) ----------
    if steps_per_day > 1:
        morning = np.exp(-0.5 * ((hour - 8.0) / 2.2) ** 2)
        evening = np.exp(-0.5 * ((hour - 19.0) / 2.6) ** 2)
        night = 0.45  # residual nighttime emissions
        diurnal_emit = night + (morning + 0.9 * evening)
        diurnal_emit = diurnal_emit / diurnal_emit.mean()  # normalize to ~1 on avg
        # A signed diurnal phase in [-1,1] for wind/PBL daytime enhancement.
        diurnal_phase = np.sin(2 * np.pi * (hour - 9.0) / 24.0)
    else:
        diurnal_emit = np.ones(n)
        diurnal_phase = np.zeros(n)

    # --- Wind, boundary layer, episodes ------------------------------------------
    wind = _wind_series(n, rng, p, diurnal_phase)
    boundary_layer = _boundary_layer(temperature, hour, wind, p)
    episode = _stagnation_episodes(n, rng, p)

    # --- Source / emission term --------------------------------------------------
    weekend_mult = np.where(is_weekend > 0, p.weekend_factor, 1.0)
    # Seasonal heating: more combustion in cold months (cool-season peak).
    heating = 1.0 + p.heating_amp * np.clip(
        np.cos(2 * np.pi * (doy - 15.0) / 365.25), -1, 1
    ) * 0.5 + p.heating_amp * 0.5
    emission = (
        p.base_emission
        * (1.0 + p.diurnal_amp * (diurnal_emit - 1.0))
        * weekend_mult
        * heating
        * (1.0 + p.stagnation_boost * episode)
    )

    # --- Ventilation (wind) and dilution (boundary layer / inversion) ------------
    ventilation = 1.0 / (1.0 + p.wind_clean * (wind / p.wind_mean))
    dilution = (p.pbl_ref / boundary_layer) ** p.inversion_strength
    pm_raw = emission * ventilation * dilution

    # --- Residual self-persistence (AR(1) stickiness) + observation noise --------
    pm = np.empty(n)
    pm[0] = pm_raw[0]
    a = p.persistence_ar1
    for i in range(1, n):
        pm[i] = a * pm[i - 1] + (1 - a) * pm_raw[i]
    noise = rng.normal(1.0, p.obs_noise_cv, size=n)
    pm25 = np.clip(pm * noise, 0.0, None)

    return pd.DataFrame(
        {
            "pm25": pm25,
            "wind": wind,
            "temp": temperature,
            "boundary_layer": boundary_layer,
            "hour": hour,
            "dow": dow,
            "is_weekend": is_weekend,
        },
        index=index,
    ).rename_axis("date")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    df = synthetic_pm25(n_days=120, seed=0, freq="h")
    print(df.describe())
    q = df["pm25"].to_numpy()
    print("pm25 min/mean/max:", float(q.min()), float(q.mean()), float(q.max()))
    print("corr(pm25, wind)          :", float(np.corrcoef(q, df["wind"])[0, 1]))
    print("corr(pm25, boundary_layer):", float(np.corrcoef(q, df["boundary_layer"])[0, 1]))
    thr = exceedance_threshold(q, 0.90)
    print(f"exceedance threshold (p90): {thr:.1f}  frac>thr={float(np.mean(q > thr)):.3f}")
