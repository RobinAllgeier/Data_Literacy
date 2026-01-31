# src/preprocess.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    ISSUE_COL,
    RETURN_COL,
    LOAN_DURATION_COL,
    DAYS_LATE_COL,
    LATE_COL,
    EXTENSIONS_COL,
    CLOSED_DATE_COL,
    LIB_WEEKMASK,
    BASE_ALLOWED_OPEN_DAYS,
    MAX_EXTENSIONS_CAP,
    USER_CATEGORY_COL,
    SOURCE_YEAR_COL,
    REMOVE_USER_CATEGORIES,
)


def _get_year_series(df: pd.DataFrame) -> pd.Series:
    """
    Returns a nullable Int64 year series for df.

    Priority:
    1) SOURCE_YEAR_COL if present
    2) year derived from ISSUE_COL (requires datetime)
    """
    if SOURCE_YEAR_COL in df.columns:
        return pd.to_numeric(df[SOURCE_YEAR_COL], errors="coerce").astype("Int64")

    if ISSUE_COL in df.columns and pd.api.types.is_datetime64_any_dtype(df[ISSUE_COL]):
        return df[ISSUE_COL].dt.year.astype("Int64")

    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


def _print_removed_by_year(removed_df: pd.DataFrame, reason: str) -> None:
    """
    Print removed counts per year for a single removal step.
    """
    if removed_df.empty:
        return

    y = _get_year_series(removed_df)
    counts = (
        removed_df.assign(_year=y)
        .dropna(subset=["_year"])
        .groupby("_year")
        .size()
        .sort_index()
    )

    if counts.empty:
        print(f"[preprocess] removed per year ({reason}): year unavailable")
        return

    per_year = ", ".join(f"{int(year)}: {int(n)}" for year, n in counts.items())
    print(f"[preprocess] removed per year ({reason}): {per_year}")


def _drop_and_report(
    df: pd.DataFrame,
    remove_mask: pd.Series,
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where remove_mask is True, and print total removed + per-year breakdown.
    Returns (new_df, removed_rows_df).
    """
    remove_mask = remove_mask.fillna(False)
    if not bool(remove_mask.any()):
        return df, pd.DataFrame(columns=df.columns)

    removed_rows = df.loc[remove_mask].copy()
    df = df.loc[~remove_mask].copy()

    print(f"[preprocess] removed {len(removed_rows)} rows: {reason}")
    _print_removed_by_year(removed_rows, reason)
    return df, removed_rows


def _print_total_removed_summary(df_start: pd.DataFrame, removed_all: pd.DataFrame) -> None:
    """
    Print total removed rows per year: absolute and percent of original rows per year.
    """
    if removed_all.empty:
        print("[preprocess] total removed summary per year: none removed")
        return

    start_year = _get_year_series(df_start)
    removed_year = _get_year_series(removed_all)

    total_per_year = (
        df_start.assign(_year=start_year)
        .dropna(subset=["_year"])
        .groupby("_year")
        .size()
        .sort_index()
    )

    removed_per_year = (
        removed_all.assign(_year=removed_year)
        .dropna(subset=["_year"])
        .groupby("_year")
        .size()
        .sort_index()
    )

    print("[preprocess] total removed summary per year (count / total = percent):")
    for year in total_per_year.index:
        total = int(total_per_year.loc[year])
        removed = int(removed_per_year.get(year, 0))
        pct = (removed / total * 100.0) if total > 0 else 0.0
        print(f"  {int(year)}: {removed}/{total} ({pct:.2f}%)")


def _remove_weird_loans_using_closed_days(
    df: pd.DataFrame,
    closed_days: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flags weird loans where open business days (Tue–Sat, excluding holidays/closed days)
    exceed allowed maximum: BASE_ALLOWED_OPEN_DAYS * (1 + extensions),
    with extensions capped at MAX_EXTENSIONS_CAP.

    Removes ONLY those weird loans that are NOT late (Verspätet == False).
    Keeps weird loans that are late.

    Returns (new_df, removed_rows_df).
    """
    if EXTENSIONS_COL not in df.columns:
        print(f"[preprocess] skip weird-loan rule: missing column {EXTENSIONS_COL}")
        return df, pd.DataFrame(columns=df.columns)

    if LATE_COL not in df.columns:
        print(f"[preprocess] skip weird-loan rule: missing column {LATE_COL}")
        return df, pd.DataFrame(columns=df.columns)

    holidays = (
        pd.to_datetime(closed_days[CLOSED_DATE_COL], dayfirst=True, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .values.astype("datetime64[D]")
    )

    start = df[ISSUE_COL].dt.normalize().values.astype("datetime64[D]")
    end = df[RETURN_COL].dt.normalize().values.astype("datetime64[D]")

    valid = (start == start) & (end == end) & (end >= start)

    open_days = np.full(len(df), np.nan, dtype="float64")
    open_days[valid] = np.busday_count(
        start[valid],
        end[valid],  # counts in [start, end)
        weekmask=LIB_WEEKMASK,
        holidays=holidays,
    ).astype("float64")

    df = df.copy()
    df["open_days_leihdauer"] = open_days

    ext = (
        pd.to_numeric(df[EXTENSIONS_COL], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=MAX_EXTENSIONS_CAP)
    )
    df["max_allowed_open_days"] = BASE_ALLOWED_OPEN_DAYS * (1 + ext)

    df["weird_loan"] = (df["open_days_leihdauer"] > df["max_allowed_open_days"]).fillna(False)

    weird_mask = df["weird_loan"].fillna(False)
    n_weird = int(weird_mask.sum())

    if n_weird > 0:
        late_rate_weird = float(df.loc[weird_mask, LATE_COL].mean())
        print(f"[preprocess] weird_loan flagged: {n_weird} rows (late-share among weird: {late_rate_weird:.3f})")
    else:
        print("[preprocess] weird_loan flagged: 0 rows")

    remove_mask = weird_mask & (~df[LATE_COL].fillna(False))
    df, removed_rows = _drop_and_report(df, remove_mask, f"weird_loan & {LATE_COL} == False")

    return df, removed_rows


def preprocess_borrowings(df: pd.DataFrame, *, closed_days: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the borrowings dataset.

    Steps:
    - drop missing/invalid issue date
    - parse issue/return timestamps
    - drop missing return timestamp
    - drop return before issue
    - remove specific user categories (configured in config.py)
    - numeric sanity for loan duration / days late
    - normalize late flag
    - apply weird-loan removal based on open business days (Tue–Sat minus closed days)

    Prints per-step removed counts + per-year breakdown and a final total per-year summary
    including percentage of original rows per year.
    """
    df = df.copy()
    df_start = df.copy()
    removed_all_steps: list[pd.DataFrame] = []

    n_start = len(df)
    print(f"[preprocess] start rows: {n_start}")

    # --- drop missing issue date (before parsing) ---
    if ISSUE_COL in df.columns:
        df, removed = _drop_and_report(df, df[ISSUE_COL].isna(), f"missing {ISSUE_COL}")
        if not removed.empty:
            removed_all_steps.append(removed)

    # --- parse datetimes ---
    df[ISSUE_COL] = pd.to_datetime(df[ISSUE_COL], errors="coerce")
    df[RETURN_COL] = pd.to_datetime(df[RETURN_COL], errors="coerce")

    # --- drop invalid issue date (after parsing) ---
    df, removed = _drop_and_report(df, df[ISSUE_COL].isna(), f"invalid {ISSUE_COL}")
    if not removed.empty:
        removed_all_steps.append(removed)

    # --- drop rows without return timestamp ---
    df, removed = _drop_and_report(df, df[RETURN_COL].isna(), f"missing {RETURN_COL}")
    if not removed.empty:
        removed_all_steps.append(removed)

    # --- remove impossible return dates ---
    df, removed = _drop_and_report(df, df[RETURN_COL] < df[ISSUE_COL], "return before issue")
    if not removed.empty:
        removed_all_steps.append(removed)

    # --- remove user categories from config ---
    if USER_CATEGORY_COL in df.columns:
        cats = df[USER_CATEGORY_COL].astype(str).str.strip()
        df, removed = _drop_and_report(
            df,
            cats.isin(set(REMOVE_USER_CATEGORIES)),
            f"{USER_CATEGORY_COL} in {sorted(set(REMOVE_USER_CATEGORIES))}",
        )
        if not removed.empty:
            removed_all_steps.append(removed)
    else:
        print(f"[preprocess] skip user-category filter: missing column {USER_CATEGORY_COL}")

    # --- numeric sanity ---
    if LOAN_DURATION_COL in df.columns:
        df[LOAN_DURATION_COL] = pd.to_numeric(df[LOAN_DURATION_COL], errors="coerce")
        neg_dur = df[LOAN_DURATION_COL].notna() & (df[LOAN_DURATION_COL] < 0)
        df, removed = _drop_and_report(df, neg_dur, f"negative {LOAN_DURATION_COL}")
        if not removed.empty:
            removed_all_steps.append(removed)

    if DAYS_LATE_COL in df.columns:
        df[DAYS_LATE_COL] = pd.to_numeric(df[DAYS_LATE_COL], errors="coerce").fillna(0)
        df, removed = _drop_and_report(df, df[DAYS_LATE_COL] < 0, f"negative {DAYS_LATE_COL}")
        if not removed.empty:
            removed_all_steps.append(removed)

    # --- late flag normalization + weird-loan rule ---
    if LATE_COL in df.columns:
        df[LATE_COL] = (
            df[LATE_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "1": True,
                    "0": False,
                    "true": True,
                    "false": False,
                    "ja": True,
                    "nein": False,
                }
            )
            .fillna(False)
            .astype(bool)
        )
        print(f"[preprocess] normalized column: {LATE_COL}")

        if closed_days is not None:
            df, removed = _remove_weird_loans_using_closed_days(df, closed_days)
            if not removed.empty:
                removed_all_steps.append(removed)
        else:
            print("[preprocess] skip weird-loan rule: closed_days not provided")
    else:
        print(f"[preprocess] skip late normalization + weird-loan rule: missing column {LATE_COL}")

    # --- final total removed per-year summary (absolute + %) ---
    if removed_all_steps:
        removed_all_df = pd.concat(removed_all_steps, ignore_index=True)
        _print_total_removed_summary(df_start, removed_all_df)
    else:
        print("[preprocess] total removed summary per year: none removed")

    print(f"[preprocess] final rows: {len(df)} (removed {n_start - len(df)} total)")
    return df.reset_index(drop=True)
