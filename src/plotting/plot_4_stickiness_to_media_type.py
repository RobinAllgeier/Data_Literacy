from __future__ import annotations

from pathlib import Path
from tueplots.constants.color import rgb

import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

from src.config import (
    MEDIA_TYPE_COL,
    USER_ID_COL,
    ISSUE_SESSION_COL,
    FIRST_K_THRESHOLDS,
    MIN_USER_SESSIONS,
    MAX_SESSION_INDEX_PLOT_4,
    SESSION_INDEX_COL,
    SESSION_CATEGORY_COL,
    STICKINESS_CURVE_SMOOTHING,
)
from src.plotting.style import apply_style


def print_media_type_session_statistics(df: pd.DataFrame):
    """
    print media type specific statistics for user sessions
    """

    pass


def make_plot(
        df: pd.DataFrame,
        outpath,
) -> None:
    apply_style()
    outpath = Path(outpath) if outpath is not None else None
    t0 = time.perf_counter()

    df_plot = df.dropna(subset=[USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL]).copy()

    # --------------------------------------------------
    # Build session-level dominant category per user-session
    # --------------------------------------------------
    session_media = (
        df_plot.groupby([USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL])
        .size()
        .rename("n")
        .reset_index()
        .sort_values([USER_ID_COL, ISSUE_SESSION_COL, "n"], ascending=[True, True, False])
    )

    session_top = (
        session_media.drop_duplicates([USER_ID_COL, ISSUE_SESSION_COL])
        .rename(columns={MEDIA_TYPE_COL: SESSION_CATEGORY_COL})
        [[USER_ID_COL, ISSUE_SESSION_COL, SESSION_CATEGORY_COL]]
        .sort_values([USER_ID_COL, ISSUE_SESSION_COL])
    )

    session_top[SESSION_INDEX_COL] = (
        session_top.groupby(USER_ID_COL)[ISSUE_SESSION_COL]
        .transform(lambda s: pd.factorize(s, sort=True)[0] + 1)
    )

    session_top = session_top[session_top[SESSION_INDEX_COL] <= MAX_SESSION_INDEX_PLOT_4].copy()
    x_all = np.arange(1, MAX_SESSION_INDEX_PLOT_4 + 1)

    # --------------------------------------------------
    # Curves + Bootstrap CI (user-level)
    # --------------------------------------------------
    N_BOOT = 300
    ALPHA = 0.05
    rng = np.random.default_rng(42)

    curves: dict[int, pd.DataFrame] = {}
    cis: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _bootstrap_ci(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        U, K = mat.shape
        boot = np.full((N_BOOT, K), np.nan)
        for b in range(N_BOOT):
            idx = rng.integers(0, U, size=U)
            boot[b] = np.nanmean(mat[idx], axis=0)
        lower = np.nanquantile(boot, ALPHA / 2, axis=0)
        upper = np.nanquantile(boot, 1 - ALPHA / 2, axis=0)
        return lower, upper

    for k0 in FIRST_K_THRESHOLDS:
        base = session_top[session_top[SESSION_INDEX_COL] <= k0].copy()

        type_k0 = (
            base.groupby([USER_ID_COL, SESSION_CATEGORY_COL])
            .size()
            .rename("n")
            .reset_index()
            .sort_values([USER_ID_COL, "n"], ascending=[True, False])
            .drop_duplicates(USER_ID_COL)
            .set_index(USER_ID_COL)[SESSION_CATEGORY_COL]
            .rename(f"type_first_{k0}")
        )

        tmp = session_top.join(type_k0, on=USER_ID_COL)

        tmp = tmp.dropna(subset=[f"type_first_{k0}"]).copy()

        col_same = f"same_as_first_{k0}"
        tmp[col_same] = (tmp[SESSION_CATEGORY_COL] == tmp[f"type_first_{k0}"])

        # --- point estimate curve ---
        curve_k0 = (
            tmp.groupby(SESSION_INDEX_COL)[col_same]
            .agg(rate="mean", n_obs="size")
            .reindex(x_all)                       # ensure full index 1..MAX
            .reset_index()
            .rename(columns={"index": SESSION_INDEX_COL})
        )

        # keep only sessions after baseline window + enough observations
        curve_k0 = curve_k0[curve_k0[SESSION_INDEX_COL] >= (k0 + 1)].copy()
        curve_k0 = curve_k0[curve_k0["n_obs"] >= MIN_USER_SESSIONS].copy()

        curves[k0] = curve_k0

        # --- bootstrap CI ---
        user_session = (
            tmp.groupby([USER_ID_COL, SESSION_INDEX_COL])[col_same]
            .mean()
            .unstack(SESSION_INDEX_COL)
            .reindex(columns=x_all)
        )

        mat = user_session.to_numpy(dtype=float)
        if mat.shape[0] < 2:
            cis[k0] = (np.full_like(x_all, np.nan, dtype=float),
                       np.full_like(x_all, np.nan, dtype=float))
        else:
            cis[k0] = _bootstrap_ci(mat)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots()

    palette = [
        rgb.pn_orange,
        rgb.tue_blue,
        rgb.tue_red,
        rgb.tue_green,
        rgb.tue_violet,
        rgb.tue_gray,
    ]

    any_line = False
    for i, k0 in enumerate(FIRST_K_THRESHOLDS):
        c = curves.get(k0)
        if c is None or c.empty:
            continue
        any_line = True

        color = palette[i % len(palette)]

        x_vals = c[SESSION_INDEX_COL].to_numpy(dtype=int)

        lower, upper = cis[k0]
        ax.fill_between(
            x_vals,
            lower[x_vals - 1],
            upper[x_vals - 1],
            alpha=0.10,
            linewidth=0,
            color=color,
            zorder=1,
        )

        ax.plot(
            x_vals,
            c["rate"],
            linewidth=0.8,
            alpha=0.18,
            color=color,
            zorder=2,
        )

        y_smooth = pd.Series(c["rate"].to_numpy(), index=x_vals).rolling(
            window=STICKINESS_CURVE_SMOOTHING, center=True, min_periods=1
        ).mean()

        ax.plot(
            x_vals,
            y_smooth.values,
            linewidth=1.3,
            marker="o",
            markersize=2.0,
            color=color,
            zorder=3,
            label=f"baseline: first {k0} sessions",
        )

    ax.set_xlabel("User session index")
    ax.set_ylabel("Share of sessions matching early dominant media type")
    ax.set_title("Consistency with early dominant media type")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis="x", which="major", color="0.88", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    k_min = min(FIRST_K_THRESHOLDS)
    x_left = k_min + 1
    ax.set_xlim(x_left, MAX_SESSION_INDEX_PLOT_4)
    ax.margins(x=0)

    fig.tight_layout()


    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot4] total time: {time.perf_counter() - t0:.2f}s")
