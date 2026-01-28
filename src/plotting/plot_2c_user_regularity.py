# src/plotting/plot_2c_user_regularity.py
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
    df_user = df.dropna(subset=[USER_ID_COL, ISSUE_COL]).copy()

    df_user['date_only'] = df_user[ISSUE_COL].dt.date
    df_user = (
        df_user.groupby([USER_ID_COL, 'date_only'])[ISSUE_COL]
        .min()
        .reset_index()
    )
    # derive time features without magic strings (use config col names)
    df_user[WEEKDAY_COL] = df_user[ISSUE_COL].dt.weekday.astype(int)  # 0=Mon ... 6=Sun
    df_user[HOUR_COL] = df_user[ISSUE_COL].dt.hour.astype(int)        # 0..23

    df_filtered = df_user[~df_user[WEEKDAY_COL].isin([0, 6])].copy()  # only Tue..Sat

    # 3. Nutzer filtern (mindestens 10 Besuche)
    user_counts = df_filtered[USER_ID_COL].value_counts()
    keep_users = user_counts[user_counts >= 10].index
    df_plot = df_filtered[df_filtered[USER_ID_COL].isin(keep_users)].copy()

    # Statistik ausgeben
    print(f"user count (>= 10 visits): {df_plot[USER_ID_COL].nunique()}")
    print(f"number of borrowings: {len(df_plot)}")


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

    fig, ax = plt.subplots()
    bins = 40
    ax.hist(entropy_obs.values, bins=bins, alpha=0.7, label="Observed")

    
    # --------------------------------------------------
    # Permutation Test Logic
    # --------------------------------------------------
    N_PERMUTATIOND = 1000  
    

    print(f"start {N_PERMUTATIOND} permutations...")

    # --------------------------------------------------
    # Randomized baseline:
    # preserve each user's number of visits, but shuffle (weekday, hour) pairs globally
    # --------------------------------------------------
    
    rng = np.random.default_rng(42)
    pairs = df_plot[[WEEKDAY_COL, HOUR_COL]].to_numpy()
    all_perm_entropies = []
    for i in range(N_PERMUTATIOND):


        pairs_shuffled = pairs[rng.permutation(len(pairs))]

        df_rand = df_plot[[USER_ID_COL]].copy()
        df_rand[WEEKDAY_COL] = pairs_shuffled[:, 0]
        df_rand[HOUR_COL] = pairs_shuffled[:, 1]

        entropy_rand = user_entropy_from_weekday_hour(df_rand)
        all_perm_entropies.extend(entropy_rand.values) 
        ax.hist(entropy_rand.values, bins=bins, alpha=0.05, color='tab:orange', label="Randomized" if i == 0 else "")

    p01_threshold = np.percentile(all_perm_entropies, 1)
    ax.axvline(p01_threshold, color='red', linestyle='--', linewidth=1.5, label='1\% Percentile')
    # --------------------------------------------------
    # Plot (single figure)
    # --------------------------------------------------

    #bins = 30
    #ax.hist(entropy_obs.values, bins=bins, alpha=0.7, label="Observed")

    ax.set_title(f"User visit regularity vs. randomized baseline")
    ax.set_xlabel("Temporal entropy (weekday × hour)")
    ax.set_ylabel("Number of users")
    leg = ax.legend()

    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    #fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath)
        plt.close(fig)
    
    plt.show()
