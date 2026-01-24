# src/plotting/plot_2b_user_regularity_vs_random.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    USER_ID_COL,
    ISSUE_COL,
    WEEKDAY_COL,
    HOUR_COL,
)

from src.plotting.style import apply_style


def make_plot(df: pd.DataFrame, outpath=None) -> None:
    apply_style()
    outpath = Path(outpath) if outpath is not None else None

    # --------------------------------------------------
    # Prepare minimal temporal features (no feature pipeline)
    # --------------------------------------------------
    df_plot = df.dropna(subset=[USER_ID_COL, ISSUE_COL]).copy()

    # derive time features without magic strings (use config col names)
    df_plot[WEEKDAY_COL] = df_plot[ISSUE_COL].dt.weekday.astype(int)  # 0=Mon ... 6=Sun
    df_plot[HOUR_COL] = df_plot[ISSUE_COL].dt.hour.astype(int)        # 0..23

    # --------------------------------------------------
    # Entropy per user over (weekday, hour)
    # --------------------------------------------------
    def user_entropy_from_weekday_hour(d: pd.DataFrame) -> pd.Series:
        counts = (
            d.groupby([USER_ID_COL, WEEKDAY_COL, HOUR_COL], dropna=True)
            .size()
            .astype(float)
        )
        p = counts / counts.groupby(level=0).sum()
        ent = (-p * np.log(p)).groupby(level=0).sum()
        ent.name = "entropy"
        return ent

    entropy_obs = user_entropy_from_weekday_hour(df_plot)

    # --------------------------------------------------
    # Randomized baseline:
    # preserve each user's number of visits, but shuffle (weekday, hour) pairs globally
    # --------------------------------------------------
    rng = np.random.default_rng(42)
    pairs = df_plot[[WEEKDAY_COL, HOUR_COL]].to_numpy()
    pairs_shuffled = pairs[rng.permutation(len(pairs))]

    df_rand = df_plot[[USER_ID_COL]].copy()
    df_rand[WEEKDAY_COL] = pairs_shuffled[:, 0]
    df_rand[HOUR_COL] = pairs_shuffled[:, 1]

    entropy_rand = user_entropy_from_weekday_hour(df_rand)

    # --------------------------------------------------
    # Plot (single figure)
    # --------------------------------------------------
    fig, ax = plt.subplots()

    bins = 30
    ax.hist(entropy_obs.values, bins=bins, alpha=0.7, label="Observed")
    ax.hist(entropy_rand.values, bins=bins, alpha=0.7, label="Randomized")

    ax.set_title("User visit regularity vs. randomized baseline")
    ax.set_xlabel("Temporal entropy (weekday × hour)")
    ax.set_ylabel("Number of users")
    ax.legend()

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath)
        plt.close(fig)
    
    plt.show()
