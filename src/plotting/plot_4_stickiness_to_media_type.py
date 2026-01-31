from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import pandas as pd

from src.config import (
    ISSUE_COL,
    USER_ID_COL,
    ISSUE_SESSION_COL,
)
from src.plotting.style import apply_style


def make_plot(
        df: pd.DataFrame,
        outpath,
        *,
        category_col: str = "Medientyp",
        k_sessions_list: list[int] | None = None,
        min_obs: int = 1000,
) -> None:
    """
    Plot: Stickiness to early dominant media type.

    For each baseline window size k0 (first k0 sessions per user):
    - determine the user's dominant media type within that baseline window
      (most frequent dominant-per-session type across the first k0 sessions)
    - for later sessions k >= k0+1: compute P(type(k) == dominant type in first k0 sessions)
    - aggregate by session_index across users and plot curves.

    Input df is loan-level (one row per borrowing). This function internally
    constructs a one-row-per-user-session table using the dominant media type
    per session (= day).
    """
    apply_style()
    outpath = Path(outpath) if outpath is not None else None
    t0 = time.perf_counter()

    if k_sessions_list is None:
        k_sessions_list = [1, 2, 3, 5, 10, 20]

    # --------------------------------------------------
    # 0) Basic checks + required columns
    # --------------------------------------------------
    required = {USER_ID_COL, category_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for this plot: {missing}")

    # Ensure issue_session exists (normally created in features.py)
    df_plot = df.copy()
    if ISSUE_SESSION_COL not in df_plot.columns:
        if ISSUE_COL not in df_plot.columns:
            raise KeyError(f"Need either {ISSUE_SESSION_COL} or {ISSUE_COL} in df.")
        df_plot[ISSUE_SESSION_COL] = pd.to_datetime(df_plot[ISSUE_COL], errors="coerce").dt.floor("D")

    # Keep rows where user + session + category exist
    df_plot = df_plot.dropna(subset=[USER_ID_COL, ISSUE_SESSION_COL, category_col]).copy()

    # --------------------------------------------------
    # 1) Build session-level dominant category per user-session
    #    (one row per user per day/session)
    # --------------------------------------------------
    session_media = (
        df_plot.groupby([USER_ID_COL, ISSUE_SESSION_COL, category_col])
        .size()
        .rename("n")
        .reset_index()
        .sort_values([USER_ID_COL, ISSUE_SESSION_COL, "n"], ascending=[True, True, False])
    )

    # dominant type per (user, session)
    session_top = (
        session_media.drop_duplicates([USER_ID_COL, ISSUE_SESSION_COL])
        .rename(columns={category_col: "session_cat"})
        [[USER_ID_COL, ISSUE_SESSION_COL, "session_cat"]]
        .sort_values([USER_ID_COL, ISSUE_SESSION_COL])
    )

    # session index per user (1,2,3,...)
    session_top["session_index"] = (
        session_top.groupby(USER_ID_COL)[ISSUE_SESSION_COL]
        .transform(lambda s: pd.factorize(s, sort=True)[0] + 1)
    )

    # --------------------------------------------------
    # 2) For each k0: baseline dominant type in first k0 sessions,
    #    then stickiness in later sessions
    # --------------------------------------------------
    curves: dict[int, pd.DataFrame] = {}

    for k0 in k_sessions_list:
        base = session_top[session_top["session_index"] <= k0].copy()

        # dominant category within the baseline window per user:
        # count how often each session_cat appears as dominant across the first k0 sessions
        type_k0 = (
            base.groupby([USER_ID_COL, "session_cat"])
            .size()
            .rename("n")
            .reset_index()
            .sort_values([USER_ID_COL, "n"], ascending=[True, False])
            .drop_duplicates(USER_ID_COL)
            .set_index(USER_ID_COL)["session_cat"]
            .rename(f"type_first_{k0}")
        )

        tmp = session_top.join(type_k0, on=USER_ID_COL)

        col_same = f"same_as_first_{k0}"
        tmp[col_same] = (tmp["session_cat"] == tmp[f"type_first_{k0}"])

        curve_k0 = (
            tmp.groupby("session_index")[col_same]
            .agg(rate="mean", n_obs="size")
            .reset_index()
        )

        # keep only sessions after baseline window
        curve_k0 = curve_k0[curve_k0["session_index"] >= (k0 + 1)].copy()
        curve_k0 = curve_k0[curve_k0["n_obs"] >= min_obs].copy()

        curves[k0] = curve_k0

    # --------------------------------------------------
    # 3) Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))

    any_line = False
    for k0 in k_sessions_list:
        c = curves.get(k0)
        if c is None or c.empty:
            continue
        any_line = True
        ax.plot(
            c["session_index"],
            c["rate"],
            linewidth=1.6,
            label=f"baseline: first {k0} sessions",
        )

    ax.set_xlabel("Session index k")
    ax.set_ylabel("P(type(k) = dominant type in first k0 sessions)")
    ax.set_title("Stickiness to early dominant media type (lines start at first new session)")

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis="x", which="major", color="0.88", linewidth=0.8)

    if any_line:
        ax.legend()

    fig.tight_layout()

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot3] total time: {time.perf_counter() - t0:.2f}s")
