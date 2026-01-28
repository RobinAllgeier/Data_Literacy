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

    # Data (one row per user-session)
    df_plot = df[df[SESSION_INDEX_COL].between(1, MAX_SESSION_INDEX_PLOT)].copy()
    x = np.arange(1, MAX_SESSION_INDEX_PLOT + 1)

    df_sessions = (
        df_plot
        .dropna(subset=[USER_ID_COL, SESSION_INDEX_COL])
        .drop_duplicates(subset=[USER_ID_COL, SESSION_INDEX_COL])
    )

    # Late curve (raw + smoothed)
    late_raw = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_LATE_FLAG_COL]
        .mean()
        .reindex(x)
    )
    late_smooth = late_raw.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()

    # Bootstrap CI (user-level)
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
    mat = df_us.to_numpy()
    U = mat.shape[0]

    boot = np.full((N_BOOT, len(x)), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, U, size=U)
        boot[b] = np.nanmean(mat[idx], axis=0)

    lower = np.nanquantile(boot, ALPHA / 2, axis=0)
    upper = np.nanquantile(boot, 1 - ALPHA / 2, axis=0)

    # Extension curve (smoothed)
    ext_raw = (
        df_sessions
        .groupby(SESSION_INDEX_COL)[SESSION_EXTENSION_FLAG_COL]
        .mean()
        .reindex(x)
    )
    ext_smooth = ext_raw.rolling(
        window=LEARNING_CURVE_SMOOTHING,
        center=True,
        min_periods=1
    ).mean()

    # Global averages (reference markers)
    global_late = float(df_sessions[SESSION_LATE_FLAG_COL].mean())
    global_ext = float(df_sessions[SESSION_EXTENSION_FLAG_COL].mean())

    # Plot
    fig, ax1 = plt.subplots()

    # TU colors
    c_late = rgb.pn_orange
    c_ext = rgb.tue_blue

    ax1.fill_between(x, lower, upper, alpha=0.12, linewidth=0, color=c_late)
    ax1.plot(x, late_raw, alpha=0.20, linewidth=1, color=c_late)

    ax1.plot(
        x,
        late_smooth,
        linewidth=1.4,
        marker="o",
        markersize=2,
        color=c_late,
    )

    ax1.set_xlabel("User session index")
    ax1.set_ylabel("Late-return probability")
    ax1.set_title("Behavioral Adaptation Across Borrowing Sessions")

    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(axis="x", which="major", color="0.88", linewidth=0.8)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        ext_smooth,
        linestyle="--",
        linewidth=1.3,
        marker="o",
        markersize=2,
        color=c_ext,
    )
    ax2.set_ylabel("Extension probability")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


    # Tight horizontal spacing
    ax1.set_xlim(1, MAX_SESSION_INDEX_PLOT)
    ax1.margins(x=0)

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot1] total time: {time.perf_counter() - t0:.2f}s")
