# src/plotting/plot_1_learning_curve_session.py
from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

from src.config import (
    USER_ID_COL,
    SESSION_INDEX_COL,
    SESSION_LATE_FLAG_COL,
    SESSION_EXTENSION_FLAG_COL,
    MAX_SESSION_INDEX_PLOT,
    LEARNING_CURVE_SMOOTHING,
)
from src.plotting.style import apply_style


def make_plot(df: pd.DataFrame, outpath) -> None:
    apply_style()
    outpath = Path(outpath) if outpath is not None else None

    t0 = time.perf_counter()

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    df_plot = df[df[SESSION_INDEX_COL].between(1, MAX_SESSION_INDEX_PLOT)].copy()
    x = np.arange(1, MAX_SESSION_INDEX_PLOT + 1)

    df_sessions = (
        df_plot
        .dropna(subset=[USER_ID_COL, SESSION_INDEX_COL])
        .drop_duplicates(subset=[USER_ID_COL, SESSION_INDEX_COL])
    )

    # --------------------------------------------------
    # Late curve
    # --------------------------------------------------
    curve_raw = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_LATE_FLAG_COL]
        .mean()
        .reindex(x)
    )
    curve_smooth = curve_raw.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()

    # --------------------------------------------------
    # Bootstrap CI
    # --------------------------------------------------
    N_BOOT = 300
    ALPHA = 0.05
    rng = np.random.default_rng(42)

    df_us = (
        df_sessions
        .groupby([USER_ID_COL, SESSION_INDEX_COL])[SESSION_LATE_FLAG_COL]
        .mean()
        .unstack(SESSION_INDEX_COL)
        .reindex(columns=x)
    )

    U = len(df_us)
    boot = np.full((N_BOOT, len(x)), np.nan)
    mat = df_us.to_numpy()

    for b in range(N_BOOT):
        idx = rng.integers(0, U, size=U)
        boot[b] = np.nanmean(mat[idx], axis=0)

    lower = np.nanquantile(boot, ALPHA / 2, axis=0)
    upper = np.nanquantile(boot, 1 - ALPHA / 2, axis=0)

    # --------------------------------------------------
    # Extension rate
    # --------------------------------------------------
    extension_rate = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_EXTENSION_FLAG_COL]
        .mean()
        .reindex(x)
    )
    extension_rate_smooth = extension_rate.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()

    # --------------------------------------------------
    # Global averages
    # --------------------------------------------------
    global_late = float(df_sessions[SESSION_LATE_FLAG_COL].mean())
    global_ext = float(df_sessions[SESSION_EXTENSION_FLAG_COL].mean())

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.fill_between(x, lower, upper, alpha=0.25)
    ax1.plot(x, curve_raw, alpha=0.35)
    ax1.plot(x, curve_smooth, linewidth=2, marker="o", markersize=3)

    ax1.set_xlabel("Borrowing order (per user)")
    ax1.set_ylabel(r"Session with $\geq 1$ late item (%)")
    ax1.set_title("Session Learning Curve with Extension Rate")

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(axis="x", which="major", color="0.85", linewidth=0.8)

    # Left axis: major ticks as percent (0 decimals)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Add avg as MINOR tick (inward) with one decimal
    ax1.set_yticks([global_late], minor=True)
    ax1.set_yticklabels([f"{global_late*100:.1f}%"], minor=True)
    ax1.tick_params(axis="y", which="minor", direction="in", length=6)

    # --------------------------------------------------
    # Right axis
    # --------------------------------------------------
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        extension_rate_smooth,
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=3
    )
    ax2.set_ylabel("Sessions with extension (%)")

    # Right axis: major ticks as percent (0 decimals)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # Add avg as MINOR tick (inward) with one decimal
    ax2.set_yticks([global_ext], minor=True)
    ax2.set_yticklabels([f"{global_ext*100:.1f}%"], minor=True)
    ax2.tick_params(axis="y", which="minor", direction="in", length=6)

    ax1.set_xlim(0.5, MAX_SESSION_INDEX_PLOT + 1.3)

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot1] total time: {time.perf_counter()-t0:.2f}s")
