# src/plotting/plot_1_learning_curve_session.py
from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd
from tueplots.constants.color import rgb

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
    # Data (one row per user-session)
    # --------------------------------------------------
    df_plot = df[df[SESSION_INDEX_COL].between(1, MAX_SESSION_INDEX_PLOT)].copy()
    x = np.arange(1, MAX_SESSION_INDEX_PLOT + 1)

    df_sessions = (
        df_plot
        .dropna(subset=[USER_ID_COL, SESSION_INDEX_COL])
        .drop_duplicates(subset=[USER_ID_COL, SESSION_INDEX_COL])
    )

    # --------------------------------------------------
    # Raw per-session estimators
    # --------------------------------------------------
    late_raw = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_LATE_FLAG_COL]
        .mean()
        .reindex(x)
    )
    ext_raw = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_EXTENSION_FLAG_COL]
        .mean()
        .reindex(x)
    )

    # Smoothed curves (main signal)
    late_smooth = late_raw.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()
    ext_smooth = ext_raw.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()

    # --------------------------------------------------
    # Bootstrap CI (user-level)
    # --------------------------------------------------
    N_BOOT = 1000
    ALPHA = 0.05
    rng = np.random.default_rng(42)

    df_us_late = (
        df_sessions
        .groupby([USER_ID_COL, SESSION_INDEX_COL])[SESSION_LATE_FLAG_COL]
        .mean()
        .unstack(SESSION_INDEX_COL)
        .reindex(columns=x)
    )
    df_us_ext = (
        df_sessions
        .groupby([USER_ID_COL, SESSION_INDEX_COL])[SESSION_EXTENSION_FLAG_COL]
        .mean()
        .unstack(SESSION_INDEX_COL)
        .reindex(columns=x)
    )

    mat_late = df_us_late.to_numpy()
    mat_ext = df_us_ext.to_numpy()
    U = mat_late.shape[0]

    boot_late = np.full((N_BOOT, len(x)), np.nan)
    boot_ext = np.full((N_BOOT, len(x)), np.nan)

    for b in range(N_BOOT):
        idx = rng.integers(0, U, size=U)
        boot_late[b] = np.nanmean(mat_late[idx], axis=0)
        boot_ext[b] = np.nanmean(mat_ext[idx], axis=0)

    late_lower = np.nanquantile(boot_late, ALPHA / 2, axis=0)
    late_upper = np.nanquantile(boot_late, 1 - ALPHA / 2, axis=0)
    ext_lower = np.nanquantile(boot_ext, ALPHA / 2, axis=0)
    ext_upper = np.nanquantile(boot_ext, 1 - ALPHA / 2, axis=0)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax1 = plt.subplots()

    c_late = rgb.pn_orange
    c_ext = rgb.tue_blue

    # --- Late-return ---
    ax1.fill_between(x, late_lower, late_upper, alpha=0.10, linewidth=0, color=c_late, zorder=1)

    # background: raw
    ax1.plot(
        x, late_raw,
        linewidth=0.8,
        alpha=0.18,
        color=c_late,
        zorder=2,
    )

    # foreground: smoothed
    ax1.plot(
        x, late_smooth,
        linewidth=1.3,
        marker="o",
        markersize=2.0,
        color=c_late,
        zorder=3,
    )

    ax1.set_xlabel("User session index")
    ax1.set_ylabel("Late-return probability")
    ax1.set_title("Behavioral Adaptation Across Borrowing Sessions")

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(axis="x", which="major", color="0.88", linewidth=0.8)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    # --- Extension ---
    ax2 = ax1.twinx()
    ax2.fill_between(x, ext_lower, ext_upper, alpha=0.08, linewidth=0, color=c_ext, zorder=1)

    # background: raw
    ax2.plot(
        x, ext_raw,
        linewidth=0.8,
        alpha=0.18,
        color=c_ext,
        zorder=2,
    )

    # foreground: smoothed
    ax2.plot(
        x, ext_smooth,
        linewidth=1.3,
        marker="o",
        markersize=2.0,
        color=c_ext,
        zorder=3,
    )

    ax2.set_ylabel("Extension probability")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax1.set_xlim(1, MAX_SESSION_INDEX_PLOT)
    ax1.margins(x=0)

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot1] total time: {time.perf_counter() - t0:.2f}s")
