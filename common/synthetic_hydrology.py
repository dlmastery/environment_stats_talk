"""Synthetic rainfall-runoff (catchment hydrology) data generator.

This produces a *physically plausible* but fully synthetic daily catchment record
so the streamflow-forecasting experiment is reproducible offline (deterministic
seeds, no network, no CAMELS download). It returns a daily DataFrame with columns
``[precip, temperature, pet, streamflow]`` indexed by ``date``.

Why this generator exists
-------------------------
Rainfall-runoff is the textbook case where machine learning *genuinely* beats the
classical toolkit. Real catchments have **nonlinear memory**: today's streamflow
depends not on today's rain alone but on how wet the soil already was (antecedent
moisture), on whether precipitation fell as rain or snow (temperature), on a
nonlinear runoff threshold (a dry soil soaks up rain; a saturated soil sheds it),
and on routing lags as water travels to the gauge. A *linear* regression on
same-day and lagged precipitation cannot represent that state-dependent, nonlinear
memory, so it underfits. An **LSTM**, which carries an internal cell state across
the lookback window, can learn exactly this kind of conditional storage — the
well-known LSTM-rainfall-runoff result.

The process below is a deliberately simple **conceptual bucket / soil-moisture
model** (in the spirit of HBV / GR4J family models, but home-grown and minimal):

1. Precipitation is split into rain vs snow by a temperature threshold; snow
   accumulates in a snowpack and melts on warm days (a degree-day melt model).
2. Liquid water (rain + melt) enters a soil-moisture store ``S`` (capacity
   ``s_max``). The fraction that becomes "quick" runoff rises **nonlinearly** with
   how full the store already is: ``(S / s_max) ** beta`` — the antecedent-moisture
   / nonlinear-threshold effect that defeats a linear model.
3. The store also drains slowly as baseflow (a linear reservoir).
4. Quick runoff + baseflow are **routed** through a short unit hydrograph (a
   triangular kernel) to impose a realistic lag/recession at the gauge.

The result has strong nonlinear, state-dependent memory but is still cheap to
generate and fully deterministic.

Swapping in real data
----------------------
``REAL_DATA_NOTE`` documents how to drop in the real **CAMELS** dataset (Catchment
Attributes and Meteorology for Large-sample Studies) instead. Keep the same
DataFrame schema and every downstream script works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Reuse the calibrated temperature/precip generators from the climate module so
# the meteorological forcing looks like the rest of the repo (import the submodule
# directly, not the common package __init__).
from common.synthetic_climate import daily_temperature, daily_precipitation

__all__ = [
    "CatchmentParams",
    "synthetic_catchment",
    "REAL_DATA_NOTE",
]

REAL_DATA_NOTE = (
    "To use real data instead of this synthetic catchment, swap in CAMELS "
    "(Catchment Attributes and Meteorology for Large-sample Studies; US: Newman "
    "et al. / Addor et al., also CAMELS-GB, -CL, -BR, -AUS). Download a basin's "
    "daily forcings (precipitation, min/max temperature, potential "
    "evapotranspiration) and observed streamflow, then assemble a DataFrame with "
    "the SAME columns ['precip', 'temperature', 'pet', 'streamflow'] indexed by a "
    "'date' DatetimeIndex. Every script here (linear baseline, LSTM, runner, "
    "tests) then works unchanged. CAMELS is the standard benchmark behind the "
    "published LSTM-rainfall-runoff studies."
)


@dataclass
class CatchmentParams:
    """Conceptual bucket parameters. Defaults give a responsive mid-latitude basin
    with clear nonlinear, state-dependent memory (so a linear model underfits)."""
    s_max: float = 120.0          # soil-moisture store capacity (mm)
    beta: float = 3.0             # nonlinearity of the runoff/saturation curve (>1)
    baseflow_k: float = 0.035     # linear baseflow recession coefficient (per day)
    et_frac: float = 0.55         # fraction of PET drawn from the store when wet
    melt_factor: float = 2.6      # degree-day snowmelt factor (mm/°C/day)
    snow_temp_c: float = 0.5      # rain/snow partition temperature (°C)
    melt_temp_c: float = 0.5      # snow melts above this air temperature (°C)
    route_len: int = 5            # unit-hydrograph length (days) for routing lag
    route_peak: int = 2           # day of the routing-kernel peak (lag to peak)
    noise_cv: float = 0.04        # multiplicative observation noise (coeff. of var.)


def _pet_from_temperature(temperature: np.ndarray, doy: np.ndarray,
                          base: float = 0.6, temp_coef: float = 0.12) -> np.ndarray:
    """A simple temperature- and season-driven potential evapotranspiration (mm/day).

    A cheap Hamon/Thornthwaite-style surrogate: PET grows with temperature and is
    larger in the high-sun part of the year. Clipped at zero (no negative PET).
    """
    seasonal = 1.0 + 0.5 * np.cos(2 * np.pi * (doy - 196) / 365.25)
    pet = base * seasonal + temp_coef * np.maximum(temperature, 0.0)
    return np.clip(pet, 0.0, None)


def _triangular_kernel(length: int, peak: int) -> np.ndarray:
    """A normalized triangular unit-hydrograph routing kernel (sums to 1)."""
    length = max(int(length), 1)
    peak = int(np.clip(peak, 0, length - 1))
    k = np.zeros(length, dtype=float)
    for i in range(length):
        # rise to the peak, then recede
        k[i] = (i + 1) / (peak + 1) if i <= peak else (length - i) / (length - peak)
    k = np.clip(k, 0.0, None)
    s = k.sum()
    return k / s if s > 0 else np.ones(length) / length


def synthetic_catchment(
    n_years: int = 20,
    start: str = "2000-01-01",
    seed: int = 0,
    params: CatchmentParams | None = None,
    init_storage_frac: float = 0.4,
) -> pd.DataFrame:
    """Generate a daily synthetic catchment record.

    Parameters
    ----------
    n_years : number of years (365 days/year, matching the climate generators).
    start   : start date of the daily index.
    seed    : master seed; temperature/precip use derived sub-seeds so the forcing
              is decorrelated but fully deterministic.
    params  : :class:`CatchmentParams` (defaults give clear nonlinear memory).
    init_storage_frac : initial soil-moisture fill as a fraction of ``s_max``.

    Returns
    -------
    DataFrame indexed by ``date`` with columns
    ``['precip', 'temperature', 'pet', 'streamflow']`` (precip & streamflow in mm,
    temperature in °C, pet in mm/day). Streamflow is strictly non-negative.
    """
    p = params or CatchmentParams()
    rng = np.random.default_rng(seed)

    # --- Meteorological forcing (reuse the repo's climate generators) ---------- #
    # Mid-latitude basin: a cool annual cycle so part of winter precip is snow,
    # which the temperature-aware models can exploit and the linear model cannot.
    # Trends are switched OFF (warming=0, intensification=0): we want a *stationary*
    # forcing regime so train/val/test are exchangeable and the only thing the LSTM
    # has to learn (and generalize) is the nonlinear catchment MEMORY — not a
    # distribution shift. (Set them non-zero to study nonstationary runoff.)
    temp_df = daily_temperature(
        n_years=n_years, start=start, seed=seed + 101,
        mean_c=8.0, annual_amp_c=12.0, noise_sd_c=2.0,
        warming_c_per_decade=0.0,
    )
    precip_df = daily_precipitation(
        n_years=n_years, start=start, seed=seed + 202,
        wet_day_prob=0.38, mean_wet_mm=7.5,
        intensification_per_decade=0.0,
    )
    dates = temp_df.index
    n = len(dates)
    temperature = temp_df["t2m"].to_numpy().astype(float)
    precip = precip_df["precip"].to_numpy().astype(float)[:n]
    doy = dates.dayofyear.to_numpy().astype(float)
    pet = _pet_from_temperature(temperature, doy)

    # --- Conceptual bucket / soil-moisture process ----------------------------- #
    s = init_storage_frac * p.s_max   # soil-moisture store (mm)
    snowpack = 0.0
    quick = np.zeros(n, dtype=float)   # fast runoff generated each day (pre-routing)
    base = np.zeros(n, dtype=float)    # slow baseflow each day (pre-routing)

    for i in range(n):
        t = temperature[i]
        # 1) Rain/snow partition + snow accumulation & degree-day melt.
        is_snow = t < p.snow_temp_c
        snowfall = precip[i] if is_snow else 0.0
        rainfall = 0.0 if is_snow else precip[i]
        snowpack += snowfall
        melt = 0.0
        if t > p.melt_temp_c and snowpack > 0.0:
            melt = min(snowpack, p.melt_factor * (t - p.melt_temp_c))
            snowpack -= melt
        liquid = rainfall + melt  # water available to the soil store today

        # 2) Nonlinear, state-dependent runoff: a fuller store sheds more water.
        sat = np.clip(s / p.s_max, 0.0, 1.0)
        runoff_coef = sat ** p.beta              # nonlinear threshold (beta > 1)
        q_quick = runoff_coef * liquid
        s += liquid - q_quick

        # 3) Evapotranspiration draws down the store (limited by availability).
        et = p.et_frac * pet[i] * np.clip(s / p.s_max, 0.0, 1.0)
        s = max(s - et, 0.0)

        # Overflow if the store exceeds capacity -> extra quick runoff.
        if s > p.s_max:
            q_quick += s - p.s_max
            s = p.s_max

        # 4) Baseflow: linear reservoir draining the store slowly.
        q_base = p.baseflow_k * s
        s = max(s - q_base, 0.0)

        quick[i] = q_quick
        base[i] = q_base

    # --- Routing: convolve quick+base with a triangular unit hydrograph -------- #
    kernel = _triangular_kernel(p.route_len, p.route_peak)
    total = quick + base
    routed = np.convolve(total, kernel, mode="full")[:n]

    # Small multiplicative observation noise (kept tiny so the signal dominates).
    noise = rng.normal(1.0, p.noise_cv, size=n)
    streamflow = np.clip(routed * noise, 0.0, None)

    return pd.DataFrame(
        {
            "precip": precip,
            "temperature": temperature,
            "pet": pet,
            "streamflow": streamflow,
        },
        index=dates,
    ).rename_axis("date")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    df = synthetic_catchment(n_years=5, seed=0)
    print(df.describe())
    print("dry-day precip fraction:", float(np.mean(df['precip'].to_numpy() == 0)))
    print("streamflow min/max:", float(df['streamflow'].min()), float(df['streamflow'].max()))
    # Correlation of streamflow with same-day precip vs a wetness proxy.
    import numpy as _np
    p7 = df['precip'].rolling(7).sum().fillna(0).to_numpy()
    print("corr(Q, precip)      :", _np.corrcoef(df['streamflow'], df['precip'])[0, 1])
    print("corr(Q, precip_7d_sum):", _np.corrcoef(df['streamflow'], p7)[0, 1])
