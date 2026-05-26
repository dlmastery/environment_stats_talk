"""Plotting helpers — always headless (Agg), consistent style, returns saved path.

Import this module *before* importing pyplot anywhere else to guarantee the Agg
backend (no display needed; works over SSH / in CI / on the 4090 box headless).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["plt", "new_fig", "save", "before_after_bars"]

# House style — calm, print-friendly, colorblind-safe-ish.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

BEFORE_COLOR = "#9aa0a6"   # muted grey — the "old way"
AFTER_COLOR = "#1a73e8"    # blue — the agentic way
ACCENT = "#188038"         # green — environment accent


def new_fig(width: float = 7.0, height: float = 4.0):
    """Return (fig, ax) at a standard slide-friendly size."""
    return plt.subplots(figsize=(width, height))


def save(fig, path: str | os.PathLike) -> str:
    """Save and close a figure; create parent dirs; return the path as str."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def before_after_bars(labels, before_vals, after_vals, ylabel: str, title: str,
                      path: str | os.PathLike, lower_is_better: bool = True) -> str:
    """A standard grouped bar chart contrasting BEFORE vs AFTER per metric/task."""
    import numpy as np
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = new_fig(max(6.0, 1.4 * len(labels)), 4.2)
    ax.bar(x - w / 2, before_vals, w, label="Before (traditional)", color=BEFORE_COLOR)
    ax.bar(x + w / 2, after_vals, w, label="After (Claude Code + AI)", color=AFTER_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{title}  ({direction})")
    ax.legend(frameon=False)
    return save(fig, path)
