"""BEFORE — the "by hand" extremes & trend workflow.

This is the kind of script an environmental statistician codes over the better
part of a day: load a precipitation series, compute the standard ETCCDI-style
extreme indices (annual maxima, Rx5day, R95p), run a Mann-Kendall trend test with
Sen's slope (implemented directly, no niche package), and read a crude return
level straight off the empirical quantiles.

It is intentionally *minimal*: point estimates, no confidence intervals, no GEV
fit, no validation gate. Its purpose is to be the honest baseline that the AFTER
pipeline (after/agentic_pipeline.py) is contrasted against.

Run from the REPO ROOT so `import common` resolves:

    python experiments/02_extreme_value_trends/before/manual_eda.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Allow `python experiments/.../before/manual_eda.py` from the repo root by
# ensuring the repo root is importable (preferred invocation is still from the
# repo root so `import common` resolves naturally).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common import daily_precipitation  # noqa: E402


# --------------------------------------------------------------------------- #
# Extreme indices (ETCCDI-style), computed by hand from a daily precip series. #
# --------------------------------------------------------------------------- #
def annual_maxima(precip: pd.Series) -> pd.Series:
    """Annual maximum 1-day precipitation (Rx1day) — the GEV block maxima."""
    return precip.groupby(precip.index.year).max()


def rx5day(precip: pd.Series) -> pd.Series:
    """Annual maximum of the 5-day running precipitation total (Rx5day)."""
    roll5 = precip.rolling(window=5, min_periods=5).sum()
    return roll5.groupby(roll5.index.year).max()


def r95p(precip: pd.Series) -> pd.Series:
    """R95p: annual total precip falling on very wet days (> 95th wet-day pct).

    The threshold is the 95th percentile of *wet-day* amounts over the record
    (the ETCCDI convention), held fixed across years.
    """
    wet = precip[precip > 0.0]
    thr = float(np.percentile(wet.to_numpy(), 95))
    very_wet = precip.where(precip > thr, other=0.0)
    return very_wet.groupby(very_wet.index.year).sum()


# --------------------------------------------------------------------------- #
# Mann-Kendall trend test + Sen's slope, implemented directly.                #
# --------------------------------------------------------------------------- #
def mann_kendall(y: np.ndarray) -> dict:
    """Mann-Kendall trend test (no ties correction beyond the standard term).

    Returns the S statistic, the normal approximation z, a two-sided p-value,
    and the Theil-Sen (Sen's) slope estimate. This is the textbook MK test —
    we implement it so the BEFORE script needs no niche `pymannkendall` package.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    # S statistic: sum of signs of all pairwise later-minus-earlier differences.
    s = 0
    for k in range(n - 1):
        s += np.sign(y[k + 1:] - y[k]).sum()
    s = float(s)

    # Variance with a tie correction term.
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # Continuity-corrected z.
    if s > 0:
        z = (s - 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s) if var_s > 0 else 0.0
    else:
        z = 0.0
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    # Sen's slope: median of all pairwise slopes (per index step = per year here).
    slopes = []
    for k in range(n - 1):
        dt = np.arange(k + 1, n) - k
        slopes.extend(((y[k + 1:] - y[k]) / dt).tolist())
    sen_slope = float(np.median(slopes)) if slopes else float("nan")

    return {"S": s, "z": float(z), "p_value": p, "sen_slope": sen_slope, "n": n}


# --------------------------------------------------------------------------- #
# Crude return level from empirical quantiles (no distribution fit).          #
# --------------------------------------------------------------------------- #
def empirical_return_level(block_maxima: np.ndarray, return_period_years: float) -> float:
    """Return level for a T-year event read off the empirical CDF of block maxima.

    For annual block maxima the non-exceedance probability is p = 1 - 1/T; we
    interpolate the empirical quantile. This is the quick-and-dirty estimate the
    AFTER pipeline replaces with a GEV fit (and gives a CI for).
    """
    bm = np.sort(np.asarray(block_maxima, dtype=float))
    p = 1.0 - 1.0 / float(return_period_years)
    return float(np.quantile(bm, np.clip(p, 0.0, 1.0)))


def run(n_years: int = 40, seed: int = 1, intensification_per_decade: float = 0.12) -> dict:
    """Execute the full manual workflow and return a flat results dict."""
    df = daily_precipitation(
        n_years=n_years, seed=seed,
        intensification_per_decade=intensification_per_decade,
    )
    precip = df["precip"]

    amax = annual_maxima(precip)
    rx5 = rx5day(precip)
    r95 = r95p(precip)

    mk_amax = mann_kendall(amax.to_numpy())
    mk_rx5 = mann_kendall(rx5.to_numpy())
    mk_r95 = mann_kendall(r95.to_numpy())

    rl = {
        T: empirical_return_level(amax.to_numpy(), T) for T in (20, 50, 100)
    }

    return {
        "years": amax.index.to_numpy().tolist(),
        "annual_maxima": amax.to_numpy().tolist(),
        "rx5day": rx5.to_numpy().tolist(),
        "r95p": r95.to_numpy().tolist(),
        "mk_annual_maxima": mk_amax,
        "mk_rx5day": mk_rx5,
        "mk_r95p": mk_r95,
        "empirical_return_levels": rl,
    }


if __name__ == "__main__":
    res = run()
    mk = res["mk_annual_maxima"]
    print("== BEFORE: manual extremes & trend EDA ==")
    print(f"  n blocks (years): {mk['n']}")
    print(f"  Annual maxima Mann-Kendall: S={mk['S']:.0f}  z={mk['z']:.2f}  "
          f"p={mk['p_value']:.4f}  Sen slope={mk['sen_slope']:.4f} mm/yr")
    print(f"  Rx5day Sen slope: {res['mk_rx5day']['sen_slope']:.4f} mm/yr "
          f"(p={res['mk_rx5day']['p_value']:.4f})")
    print(f"  R95p   Sen slope: {res['mk_r95p']['sen_slope']:.4f} mm/yr "
          f"(p={res['mk_r95p']['p_value']:.4f})")
    print("  Empirical return levels (mm):")
    for T, v in res["empirical_return_levels"].items():
        print(f"    {T:>3}-yr: {v:7.2f}")
    print("  NOTE: point estimates only; no GEV fit, no CIs, no validation gate.")
