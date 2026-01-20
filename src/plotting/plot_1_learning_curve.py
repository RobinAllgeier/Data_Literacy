# src/plotting/plot_1_learning_curve.py
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from tueplots.constants.color import rgb

from src.config import USER_ID_COL, SESSION_INDEX_COL, LATE_FLAG_COL, MAX_SESSION_INDEX_PLOT
from src.plotting.style import apply_style


def make_plot(df: pd.DataFrame, outpath) -> None:
    apply_style()

    d = df[df[USER_ID_COL].notna()].copy()
    d = d[d[SESSION_INDEX_COL].notna()]
    d = d[d[SESSION_INDEX_COL] <= MAX_SESSION_INDEX_PLOT]

    g = d.groupby(SESSION_INDEX_COL)[LATE_FLAG_COL].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(g.index, g.values, marker="o", color=rgb.tue_blue)
    ax.set_title("Late-return rate by user experience (session index)")
    ax.set_xlabel("Session index (per user)")
    ax.set_ylabel("Late-return rate")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
