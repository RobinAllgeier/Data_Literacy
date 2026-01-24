# src/features.py
from __future__ import annotations

import pandas as pd

from src.config import (
    ISSUE_COL,
    USER_ID_COL,
    LATE_COL,
    EXTENSIONS_COL,
    EXPERIENCE_CUTOFF,
    LATE_FLAG_COL,
    ISSUE_SESSION_COL,
    SESSION_INDEX_COL,
    SESSION_SIZE_COL,
    EXPERIENCE_STAGE_COL,
    SESSION_LATE_FLAG_COL,
    SESSION_EXTENSION_FLAG_COL,
    WEEKDAY_COL,
    HOUR_COL,
    USER_MODAL_WEEKDAY_COL,
    USER_MODAL_HOUR_COL,
    USER_MATCH_TYPICAL_COL,
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analysis-ready features to the cleaned borrowings dataset.
    """
    df = df.copy()

    # --- item-level late flag ---
    df[LATE_FLAG_COL] = df[LATE_COL].astype(bool)

    # --- user-based features (only where user id exists) ---
    has_user = df[USER_ID_COL].notna()

    # issue session (calendar day)
    df.loc[has_user, ISSUE_SESSION_COL] = (
        df.loc[has_user, ISSUE_COL].dt.floor("D")
    )

    # session index per user
    df.loc[has_user, SESSION_INDEX_COL] = (
        df.loc[has_user]
        .groupby(USER_ID_COL)[ISSUE_SESSION_COL]
        .transform(lambda s: pd.factorize(s, sort=True)[0] + 1)
    )

    # session size
    df.loc[has_user, SESSION_SIZE_COL] = (
        df.loc[has_user]
        .groupby([USER_ID_COL, ISSUE_SESSION_COL])[ISSUE_SESSION_COL]
        .transform("size")
    )

    # session-level late flag
    df.loc[has_user, SESSION_LATE_FLAG_COL] = (
        df.loc[has_user]
        .groupby([USER_ID_COL, ISSUE_SESSION_COL])[LATE_FLAG_COL]
        .transform("any")
    )

    # session-level extension flag
    df.loc[has_user, SESSION_EXTENSION_FLAG_COL] = (
        df.loc[has_user]
        .groupby([USER_ID_COL, ISSUE_SESSION_COL])[EXTENSIONS_COL]
        .transform(lambda s: (s > 0).any())
    )

    # experience stage
    df.loc[has_user, EXPERIENCE_STAGE_COL] = (
        df.loc[has_user, SESSION_INDEX_COL]
        .le(EXPERIENCE_CUTOFF)
        .map({True: "early", False: "experienced"})
    )

    # --- timing features ---
    df.loc[has_user, WEEKDAY_COL] = df.loc[has_user, ISSUE_COL].dt.weekday
    df.loc[has_user, HOUR_COL] = df.loc[has_user, ISSUE_COL].dt.hour

    # per-user typical time (mode)
    user_modal = (
        df.loc[has_user, [USER_ID_COL, WEEKDAY_COL, HOUR_COL]]
        .groupby(USER_ID_COL)
        .agg(
            **{
                USER_MODAL_WEEKDAY_COL: (WEEKDAY_COL, lambda s: s.mode().iloc[0]),
                USER_MODAL_HOUR_COL: (HOUR_COL, lambda s: s.mode().iloc[0]),
            }
        )
    )

    df = df.merge(user_modal, on=USER_ID_COL, how="left")

    df[USER_MATCH_TYPICAL_COL] = (
        (df[WEEKDAY_COL] == df[USER_MODAL_WEEKDAY_COL]) &
        (df[HOUR_COL] == df[USER_MODAL_HOUR_COL])
    )

    return df
