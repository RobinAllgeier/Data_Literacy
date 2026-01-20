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
)

def _remove_weird_loans_using_closed_days(df: pd.DataFrame, closed_days: pd.DataFrame) -> pd.DataFrame:
    """
    Flags weird loans where open business days (Tue–Sat, excluding holidays/closed days)
    exceed allowed maximum: BASE_ALLOWED_OPEN_DAYS * (1 + extensions),
    with extensions capped at MAX_EXTENSIONS_CAP.

    Removes ONLY those weird loans that are NOT late (Verspätet == False).
    Keeps weird loans that are late.
    Prints:
      - how many weird loans exist
      - share of late among weird loans
      - how many rows removed (weird & not late)
    """
    if EXTENSIONS_COL not in df.columns:
        print(f"[preprocess] skip weird-loan rule: missing column {EXTENSIONS_COL}")
        return df

    if LATE_COL not in df.columns:
        print(f"[preprocess] skip weird-loan rule: missing column {LATE_COL}")
        return df

    # closed days -> holidays array in datetime64[D]
    holidays = (
        pd.to_datetime(closed_days[CLOSED_DATE_COL], dayfirst=True, errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .values.astype("datetime64[D]")
    )

    start = df[ISSUE_COL].dt.normalize().values.astype("datetime64[D]")
    end = df[RETURN_COL].dt.normalize().values.astype("datetime64[D]")

    # valid rows for busday_count (NaT-safe)
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

    # allowed max = BASE_ALLOWED_OPEN_DAYS * (1 + extensions), extensions capped
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

    # REMOVE ONLY weird loans that are NOT late
    remove_mask = weird_mask & (~df[LATE_COL].fillna(False))

    before = len(df)
    df = df.loc[~remove_mask].copy()
    removed = before - len(df)

    if removed > 0:
        print(f"[preprocess] removed {removed} rows: weird_loan & {LATE_COL} == False")
    else:
        print(f"[preprocess] removed 0 rows: weird_loan & {LATE_COL} == False")

    return df


def preprocess_borrowings(df: pd.DataFrame, *, closed_days: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the borrowings dataset.

    Removes records with invalid or inconsistent dates, normalizes the
    late-return flag, and applies library-specific rules based on open
    business days and extension limits. Prints summary information about
    removed rows for transparency.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw borrowings data.

    closed_days : pandas.DataFrame
        Library closed days used to validate loan durations.

    Returns
    -------
    pandas.DataFrame
        Cleaned borrowings data.
    """
    df = df.copy()
    n_start = len(df)
    print(f"[preprocess] start rows: {n_start}")

    # --- drop missing issue date ---
    before = len(df)
    df = df.dropna(subset=[ISSUE_COL])
    removed = before - len(df)
    if removed > 0:
        print(f"[preprocess] removed {removed} rows: missing {ISSUE_COL}")

    # --- parse datetimes ---
    df[ISSUE_COL] = pd.to_datetime(df[ISSUE_COL], errors="coerce")
    df[RETURN_COL] = pd.to_datetime(df[RETURN_COL], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[ISSUE_COL])
    removed = before - len(df)
    if removed > 0:
        print(f"[preprocess] removed {removed} rows: invalid {ISSUE_COL}")

    # --- drop rows without return timestamp ---
    before = len(df)
    df = df.dropna(subset=[RETURN_COL])
    removed = before - len(df)
    if removed > 0:
        print(f"[preprocess] removed {removed} rows: missing {RETURN_COL}")

    # --- remove impossible return dates ---
    before = len(df)
    df = df[df[RETURN_COL] >= df[ISSUE_COL]]
    removed = before - len(df)
    if removed > 0:
        print(f"[preprocess] removed {removed} rows: return before issue")

   # --- numeric sanity ---
    if LOAN_DURATION_COL in df.columns:
        before = len(df)
        df[LOAN_DURATION_COL] = pd.to_numeric(df[LOAN_DURATION_COL], errors="coerce")
        df = df[df[LOAN_DURATION_COL].isna() | (df[LOAN_DURATION_COL] >= 0)]
        removed = before - len(df)
        if removed:
            print(f"[preprocess] removed {removed} rows: negative {LOAN_DURATION_COL}")

        if DAYS_LATE_COL in df.columns:
            df[DAYS_LATE_COL] = pd.to_numeric(df[DAYS_LATE_COL], errors="coerce").fillna(0)
            before = len(df)
            df = df[df[DAYS_LATE_COL] >= 0]
            removed = before - len(df)
            if removed:
                print(f"[preprocess] removed {removed} rows: negative {DAYS_LATE_COL}")


    # --- late flag normalization ---
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

        # --- apply library-rule weird-loan removal ---
        if closed_days is not None:
            df = _remove_weird_loans_using_closed_days(df, closed_days)
        else:
            print("[preprocess] skip weird-loan rule: closed_days not provided")

    print(f"[preprocess] final rows: {len(df)} (removed {n_start - len(df)} total)")
    return df.reset_index(drop=True)
