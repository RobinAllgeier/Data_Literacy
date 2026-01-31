# src/plotting/plot_2a_library_visit_clock.py
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots.constants.color import rgb

from src.plotting.style import apply_style
from src.config import ISSUE_COL, USER_ID_COL, SESSION_INDEX_COL, USER_STD_HOUR_COL, WEEKDAY_COL, USER_MODAL_WEEKDAY_COL

def print_user_statistics(df: pd.DataFrame):
    """
    Calculate time-based statistics at user level
    """
   
    is_tue_to_sat = df[WEEKDAY_COL].between(1, 5)  # Tue–Sat
    user_sessions = df.groupby(USER_ID_COL)[SESSION_INDEX_COL].max()
    frequent_user = user_sessions[user_sessions >= 10].index        # user with at least 10 sessions

    df_user = df[is_tue_to_sat & df[USER_ID_COL].isin(frequent_user)].copy()

    # --- weekday preference ---
    df_user["is_modal_day"] = (df_user[WEEKDAY_COL] == df_user[USER_MODAL_WEEKDAY_COL])
    user_weekday_probs = df_user.groupby(USER_ID_COL)["is_modal_day"].mean()
    weekday_prob = user_weekday_probs.mean() * 100

    # --- standard deviation of the time ---
    user_std_devs = df_user.groupby(USER_ID_COL)[USER_STD_HOUR_COL].first()
    mean_std_dev = user_std_devs.mean()

    # --- user with std < 1 ---
    precise_users_mask = user_std_devs < 1.0
    precise_percentage = precise_users_mask.mean() * 100

    print(f"[stats] user with >= 10 sessions, n={len(frequent_user)}):")
    print(f"  1. Mean probability of returning on the same weekday: {weekday_prob:.1f}%")
    print(f"  2. Mean standard deviation of visit times: {mean_std_dev:.1f} hours")
    print(f"  3. Users with high temporal precision (<1h std): {precise_percentage:.1f}%")

def time_to_theta(minutes: np.ndarray) -> np.ndarray:
    """
    Nonlinear clock mapping:
    - compress 20:00–09:00
    - expand 09:00–20:00
    Returned angles are in [0, 2pi) but NOT guaranteed monotonic over 0..1440.
    """
    minutes = np.asarray(minutes, dtype=float)

    night_start = 20 * 60  # 20:00
    night_end = 9 * 60     # 09:00

    night_span = (24 * 60 - night_start) + night_end   # 13h
    day_span = night_start - night_end                 # 11h

    night_frac = 0.12
    night_angle = 2 * np.pi * night_frac
    day_angle = 2 * np.pi - night_angle

    theta = np.zeros_like(minutes, dtype=float)

    # Night: [20:00, 24:00)
    mask_n1 = minutes >= night_start
    theta[mask_n1] = ((minutes[mask_n1] - night_start) / night_span) * night_angle

    # Night: [00:00, 09:00)
    mask_n2 = minutes < night_end
    theta[mask_n2] = ((minutes[mask_n2] + (24 * 60 - night_start)) / night_span) * night_angle

    # Day: [09:00, 20:00)
    mask_day = (~mask_n1) & (~mask_n2)
    theta[mask_day] = night_angle + ((minutes[mask_day] - night_end) / day_span) * day_angle

    return theta


def _bin_overlaps_interval(bin_start: np.ndarray, bin_end: np.ndarray, open_min: int, close_min: int) -> np.ndarray:
    """True if [bin_start, bin_end) overlaps with [open_min, close_min)."""
    return (bin_end > open_min) & (bin_start < close_min)


def _unwrap_theta_edges(theta_edges: np.ndarray) -> np.ndarray:
    """Make theta edges strictly increasing around the circle so polar bars render correctly."""
    te = np.asarray(theta_edges, dtype=float)
    te_u = np.unwrap(te)          # remove jumps of +-2pi
    te_u = te_u - te_u[0]         # start at 0
    te_u = te_u * (2 * np.pi / te_u[-1])  # scale to exactly one turn
    return te_u

def bar_with_cap(ax, theta_starts, heights, widths, base_rgb, *,
                 bottom=None, fill_alpha=0.35, cap_frac=0.01,
                 zorder=1.0, label=None):
    """
    Draw polar bars with transparent fill + opaque cap at the top.
    Supports a per-bin 'bottom' (array-like) to stack / cut-out radially.
    """
    heights = np.asarray(heights, dtype=float)
    if bottom is None:
        bottom = np.zeros_like(heights)
    else:
        bottom = np.asarray(bottom, dtype=float)

    # transparent fill
    ax.bar(
        theta_starts,
        heights,
        width=widths,
        bottom=bottom,
        align="edge",
        color=(*base_rgb, fill_alpha),
        edgecolor="none",
        linewidth=0.0,
        zorder=zorder,
        label=label,
    )

    # opaque cap (only where height > 0)
    cap_h = cap_frac * ax.get_rmax()
    cap_h = max(cap_h, 1e-9)

    cap = np.minimum(heights, cap_h)
    cap_bottom = bottom + (heights - cap)

    mask = heights > 0
    ax.bar(
        np.asarray(theta_starts)[mask],
        np.asarray(cap)[mask],
        width=np.asarray(widths)[mask],
        bottom=np.asarray(cap_bottom)[mask],
        align="edge",
        color=(*base_rgb, 1.0),
        edgecolor="none",
        linewidth=0.0,
        zorder=zorder + 0.05,
    )




def make_plot(df: pd.DataFrame, outpath) -> None:
    apply_style()
    #plt.rcParams.update(bundles.icml2024(column="half", nrows=1, ncols=1))

    outpath = Path(outpath) if outpath is not None else None
    t0 = time.perf_counter()

    # -----------------------------
    # Prepare
    # -----------------------------
    df_plot = df.dropna(subset=[USER_ID_COL, ISSUE_COL]).copy()
    if not np.issubdtype(df_plot[ISSUE_COL].dtype, np.datetime64):
        df_plot[ISSUE_COL] = pd.to_datetime(df_plot[ISSUE_COL], errors="coerce")
        df_plot = df_plot.dropna(subset=[ISSUE_COL])

    df_plot["weekday"] = df_plot[ISSUE_COL].dt.weekday  # Mon=0..Sun=6
    df_plot["date"] = df_plot[ISSUE_COL].dt.floor("D")
    df_plot["minute_of_day"] = df_plot[ISSUE_COL].dt.hour * 60 + df_plot[ISSUE_COL].dt.minute
    OPEN = 10 * 60 + 30      # 10:30
    CLOSE_TUE_FRI = 19 * 60  # 19:00
    CLOSE_SAT = 14 * 60      # 14:00

    m = df_plot["minute_of_day"]
    wd = df_plot["weekday"]  # Mon=0..Sun=6

    # Tue–Fri: 10:30–19:00
    keep_tue_fri = wd.between(1, 4) & (m >= OPEN) & (m < CLOSE_TUE_FRI)

    # Saturday: 10:30–14:00o
    keep_sat = (wd == 5) & (m >= OPEN) & (m < CLOSE_SAT)

    df_plot = df_plot[keep_tue_fri | keep_sat].copy()

    df_plot["bin_30m"] = (df_plot["minute_of_day"] // 30).astype(int)  # 0..47

    daily_bin_users = (
        df_plot
        .groupby(["date", "weekday", "bin_30m"])[USER_ID_COL]
        .nunique()
        .reset_index(name="n_users")
    )

    tue_fri = (
        daily_bin_users[daily_bin_users["weekday"].between(1, 4)]
        .groupby("bin_30m")["n_users"]
        .mean()
        .reindex(range(48), fill_value=0.0)
        .to_numpy()
    )

    sat = (
        daily_bin_users[daily_bin_users["weekday"] == 5]
        .groupby("bin_30m")["n_users"]
        .mean()
        .reindex(range(48), fill_value=0.0)
        .to_numpy()
    )

    # -----------------------------
    # Theta geometry (UNWRAPPED edges)
    # -----------------------------
    bin_edges_min = np.arange(49) * 30
    theta_edges_raw = time_to_theta(bin_edges_min)
    theta_edges = _unwrap_theta_edges(theta_edges_raw)

    theta_starts = theta_edges[:-1]
    theta_widths = np.diff(theta_edges)
    theta_centers = theta_starts + 0.5 * theta_widths

    bin_start_min = bin_edges_min[:-1]
    bin_end_min = bin_edges_min[1:]

    # -----------------------------
    # Closed-time rings
    # Tue–Fri open 10:30–19:00
    # Sat open    10:30–14:00
    # -----------------------------
    OPEN = 10 * 60 + 30
    CLOSE_TUE_FRI = 19 * 60
    CLOSE_SAT = 14 * 60

    open_tue_fri = _bin_overlaps_interval(bin_start_min, bin_end_min, OPEN, CLOSE_TUE_FRI)
    open_sat = _bin_overlaps_interval(bin_start_min, bin_end_min, OPEN, CLOSE_SAT)

    closed_tue_fri = ~open_tue_fri
    closed_sat = ~open_sat

    # -----------------------------
    # Plot
    # -----------------------------
    figwidth = plt.rcParams.get("figure.figsize")[0]  # halbe Breite von tueplots
    fig = plt.figure(figsize=(figwidth, figwidth))     # quadratisch
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    ax.set_title("Average Library Usage by Time of Day")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    r_max = float(max(np.max(tue_fri), np.max(sat), 1.0))
    ax.set_rlim(0, r_max * 1.15)

    # Cut out only the overlapping *part* (radially):
    # Saturday is shown only where it exceeds Tue–Fri, and only above Tue–Fri.
    sat_above = np.maximum(0.0, sat - tue_fri)
    sat_bottom = np.minimum(sat, tue_fri)  # start Saturday where Tue–Fri ends

    
    # Saturday: only the part ABOVE Tue–Fri, starting at Tue–Fri height
    bar_with_cap(
        ax, theta_starts, sat_above, theta_widths, rgb.pn_orange,
        bottom=sat_bottom,
        fill_alpha=0.20, cap_frac=0.02, zorder=1.0, label="Saturday"
    )

    # Tue–Fri: normal from zero, on top
    bar_with_cap(
        ax, theta_starts, tue_fri, theta_widths, rgb.tue_blue,
        bottom=None,
        fill_alpha=0.30, cap_frac=0.02, zorder=1.2, label="Tue–Fri"
    )


    # --- filled half-hour "bars" (sectors) from center to value ---
    # Draw Tue–Fri first, then Saturday with transparent orange on top
    # Use matching edgecolor to avoid gaps between bars
    # ax.bar(
    #     theta_starts,
    #     sat,
    #     width=theta_widths,
    #     bottom=0.0,
    #     align="edge",
    #     color=rgb.pn_orange,
    #     alpha=1.0,
    #     edgecolor=rgb.pn_orange,
    #     linewidth=0.5,
    #     zorder=1,
    #     label="Saturday"
    # )
    
    # ax.bar(
    #     theta_starts,
    #     tue_fri,
    #     width=theta_widths,
    #     bottom=0.0,
    #     align="edge",
    #     color=rgb.tue_blue,
    #     alpha=1.0,
    #     edgecolor=rgb.tue_blue,
    #     linewidth=0.5,
    #     zorder=1.1,
    #     label="Tue–Fri"
    # )

    # r_max = float(max(np.max(tue_fri), np.max(sat), 1.0))
    # ax.set_rlim(0, r_max * 1.15)

    # Grey rings for CLOSED times (outer = Sat, inner = Tue–Fri)
    # Group consecutive closed bins to avoid gaps
    def draw_closed_rings(closed_mask, bottom, height, alpha):
        i = 0
        while i < len(closed_mask):
            if closed_mask[i]:
                # Find end of consecutive closed bins
                j = i
                while j < len(closed_mask) and closed_mask[j]:
                    j += 1
                # Draw one bar from theta_starts[i] to theta_edges[j]
                total_width = theta_edges[j] - theta_starts[i]
                ax.bar(
                    float(theta_starts[i]),
                    height,
                    width=float(total_width),
                    bottom=bottom,
                    align="edge",
                    color="lightgray",
                    edgecolor="none",
                    alpha=alpha,
                    zorder=0,
                )
                i = j
            else:
                i += 1

    outer_bottom = r_max * 1.06
    outer_height = r_max * 0.07
    draw_closed_rings(closed_sat, outer_bottom, outer_height, 1.0)

    inner_bottom = r_max * 0.94
    inner_height = r_max * 0.07
    draw_closed_rings(closed_tue_fri, inner_bottom, inner_height, 0.7)

    # Ticks: map minutes -> raw theta -> unwrap consistently
    tick_minutes = np.array(
        [9 * 60, 10 * 60 + 30, 11 * 60, 12 * 60, 14 * 60, 16 * 60, 18 * 60, 19 * 60, 20 * 60, 0],
        dtype=float,
    )
    tick_theta_raw = time_to_theta(tick_minutes)
    tick_theta = np.unwrap(tick_theta_raw - theta_edges_raw[0])
    tick_theta = tick_theta * (2 * np.pi / (np.unwrap(theta_edges_raw - theta_edges_raw[0])[-1]))
    tick_labels = ["09:00", "10:30", "11:00", "12:00", "14:00", "16:00", "18:00", "19:00", "20:00", "00:00"]
    ax.set_xticks(tick_theta)
    ax.set_xticklabels(tick_labels)

    #ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.08), frameon=False)
    # ax.text(
    #     0.5, -0.12,
    #     "Grey rings: closed times (outer = Sat, inner = Tue–Fri)\n"
    #     "Clock angles are nonlinearly scaled to compress 20:00–09:00.",
    #     transform=ax.transAxes,
    #     ha="center", va="top",
    # )

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"[plot2a] total time: {time.perf_counter() - t0:.2f}s")
