# src/features.py
from __future__ import annotations

import pandas as pd

from src.config import (
    ISSUE_COL,
    USER_ID_COL,
    LATE_COL,
    EXPERIENCE_CUTOFF,
    LATE_FLAG_COL,
    ISSUE_SESSION_COL,
    SESSION_INDEX_COL,
    SESSION_SIZE_COL,
    EXPERIENCE_STAGE_COL,
)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add analysis-ready features to the cleaned borrowings dataset.
    """
    df = df.copy()

    # --- final late flag ---
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

    # experience stage
    df.loc[has_user, EXPERIENCE_STAGE_COL] = (
        df.loc[has_user, SESSION_INDEX_COL]
        .le(EXPERIENCE_CUTOFF)
        .map({True: "early", False: "experienced"})
    )

    return df
