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


def print_media_type_session_statistics(df: pd.DataFrame) -> None:
    """
    Print media type specific statistics for user sessions:
    - tie-session rate (no unique dominant media type)
    - distribution of number of distinct media types per session (1..10)
    """
    df_s = df.dropna(subset=[USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL]).copy()

    counts = (
        df_s.groupby([USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL])
        .size()
        .rename("n")
        .reset_index()
    )

    # ----------------------------
    # Overall media type distribution (loan-level)
    # ----------------------------
    mt = df_s[MEDIA_TYPE_COL].astype(str).str.strip()
    dist = (
        mt.value_counts(normalize=True, dropna=False)
        .mul(100.0)
        .round(2)
        .rename("pct")
        .reset_index()
        .rename(columns={"index": MEDIA_TYPE_COL})
    )

    print("[plot4][stats] overall media type distribution (loan-level, %)")
    print(dist.to_string(index=False))

    # ----------------------------
    # Tie sessions (dominant not unique)
    # ----------------------------
    max_per_session = (
        counts.groupby([USER_ID_COL, ISSUE_SESSION_COL])["n"]
        .max()
        .rename("max_n")
        .reset_index()
    )

    counts2 = counts.merge(max_per_session, on=[USER_ID_COL, ISSUE_SESSION_COL], how="left")
    counts2["is_max"] = counts2["n"].eq(counts2["max_n"])

    n_at_max = (
        counts2.groupby([USER_ID_COL, ISSUE_SESSION_COL])["is_max"]
        .sum()
        .rename("n_tied_at_max")
    )

    n_sessions = int(n_at_max.shape[0])
    n_tie_sessions = int((n_at_max > 1).sum())
    tie_rate = (n_tie_sessions / n_sessions * 100.0) if n_sessions else 0.0

    print(f"[plot4][stats] tie sessions: {n_tie_sessions}/{n_sessions} ({tie_rate:.2f}%)")

    print("[plot4][stats] baseline tie-rate by k0 (MERGED borrowings across first k0 sessions):")
    for k0 in FIRST_K_THRESHOLDS:
        n_users, n_tied, tie_rate = _baseline_tie_rate_merged_borrowings(df_s, k0)
        print(f"  k0={k0:>2}: {n_tied}/{n_users} users tied ({tie_rate:.2f}%)")

# ----------------------------
    # #distinct media types per session: P(m = 1..10)
    # ----------------------------
    n_types = (
        counts.groupby([USER_ID_COL, ISSUE_SESSION_COL])[MEDIA_TYPE_COL]
        .nunique()
        .rename("n_media_types")
    )

    max_k = 10
    pmf = (
        n_types.value_counts(normalize=True)
        .sort_index()
    )

    idx = pd.Index(range(1, max_k + 1), name="n_media_types")
    pmf_1_10 = pmf.reindex(idx, fill_value=0.0).rename("probability").reset_index()

    p_gt_10 = float(pmf[pmf.index > max_k].sum()) if (pmf.index > max_k).any() else 0.0

    print("[plot4][stats] distribution: #media types per session (probability)")
    pmf_print = pmf_1_10.copy()
    pmf_print["probability"] = (pmf_print["probability"] * 100.0).round(2)
    print(pmf_print.to_string(index=False))

    if p_gt_10 > 0:
        print(f"[plot4][stats] P(n_media_types > {max_k}) = {p_gt_10*100.0:.2f}%")



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
    # (DROP TIES: sessions without a unique dominant media type)
    # --------------------------------------------------
    session_top = _get_prepared_session_data(df_plot)
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
        base_loans = df_plot[df_plot[SESSION_INDEX_COL] <= k0].dropna(subset=[USER_ID_COL, MEDIA_TYPE_COL]).copy()

        uc = (
            base_loans.groupby([USER_ID_COL, MEDIA_TYPE_COL])
            .size()
            .rename("n")
            .reset_index()
        )

        uc["max_n"] = uc.groupby(USER_ID_COL)["n"].transform("max")
        uc["is_max"] = uc["n"].eq(uc["max_n"])
        uc["n_tied_at_max"] = uc.groupby(USER_ID_COL)["is_max"].transform("sum")

        type_k0 = (
            uc[(uc["is_max"]) & (uc["n_tied_at_max"] == 1)]
            .set_index(USER_ID_COL)[MEDIA_TYPE_COL]
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
            .reindex(x_all)
            .reset_index()
            .rename(columns={"index": SESSION_INDEX_COL})
        )

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

def _get_prepared_session_data(input_data: pd.DataFrame) -> pd.DataFrame:
    # input_data ist loan-level und hat SESSION_INDEX_COL schon aus add_features
    df_plot = input_data.dropna(subset=[USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL, SESSION_INDEX_COL]).copy()

    # session_index pro (user, session) aus dem Input übernehmen (konstant pro Session)
    session_idx = (
        df_plot.groupby([USER_ID_COL, ISSUE_SESSION_COL])[SESSION_INDEX_COL]
        .first()
        .reset_index()
    )

    # counts pro (user, session, media_type)
    session_media = (
        df_plot.groupby([USER_ID_COL, ISSUE_SESSION_COL, MEDIA_TYPE_COL])
        .size()
        .rename("n")
        .reset_index()
    )

    # tie detection innerhalb der Session
    session_media["n_max"] = session_media.groupby([USER_ID_COL, ISSUE_SESSION_COL])["n"].transform("max")
    session_media["is_max"] = session_media["n"].eq(session_media["n_max"])
    session_media["n_tied_at_max"] = session_media.groupby([USER_ID_COL, ISSUE_SESSION_COL])["is_max"].transform("sum")

    dom = session_media[(session_media["is_max"]) & (session_media["n_tied_at_max"] == 1)].copy()

    session_top = (
        dom.rename(columns={MEDIA_TYPE_COL: SESSION_CATEGORY_COL})
        [[USER_ID_COL, ISSUE_SESSION_COL, SESSION_CATEGORY_COL]]
        .merge(session_idx, on=[USER_ID_COL, ISSUE_SESSION_COL], how="left")
        .sort_values([USER_ID_COL, SESSION_INDEX_COL])
    )

    session_top = session_top[session_top[SESSION_INDEX_COL] <= MAX_SESSION_INDEX_PLOT_4].copy()
    return session_top

def _baseline_tie_rate_merged_borrowings(df_loans: pd.DataFrame, k0: int) -> tuple[int, int, float]:
    df0 = df_loans.dropna(subset=[USER_ID_COL, SESSION_INDEX_COL, MEDIA_TYPE_COL]).copy()

    # eligible users: have at least k0 sessions overall
    max_sess = df0.groupby(USER_ID_COL)[SESSION_INDEX_COL].max()
    eligible = max_sess[max_sess >= k0].index
    n_users = int(len(eligible))
    if n_users == 0:
        return 0, 0, 0.0

    base = df0[df0[USER_ID_COL].isin(eligible) & (df0[SESSION_INDEX_COL] <= k0)].copy()
    if base.empty:
        return n_users, 0, 0.0

    uc = (
        base.groupby([USER_ID_COL, MEDIA_TYPE_COL])
        .size()
        .rename("n")
        .reset_index()
    )

    uc["max_n"] = uc.groupby(USER_ID_COL)["n"].transform("max")
    uc["is_max"] = uc["n"].eq(uc["max_n"])
    n_at_max = uc.groupby(USER_ID_COL)["is_max"].sum()

    n_tied = int((n_at_max > 1).sum())
    tie_rate = (n_tied / n_users * 100.0) if n_users else 0.0
    return n_users, n_tied, tie_rate

