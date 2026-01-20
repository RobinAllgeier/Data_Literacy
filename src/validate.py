# src/validate.py
from __future__ import annotations

import pandas as pd

from src.config import (
    ISSUE_COL,
    RETURN_COL,
    USER_ID_COL,
    LATE_COL,
    LATE_FLAG_COL,
    SESSION_INDEX_COL,
)

def validate_borrowings(df: pd.DataFrame) -> None:
    """
    Validate core invariants of the cleaned + feature-enriched borrowings data.
    Raises AssertionError if a violation is detected.
    """

    # --- date consistency ---
    assert df[ISSUE_COL].notna().all(), "Missing issue dates remain"
    assert df[RETURN_COL].notna().all(), "Missing return dates remain"
    assert (df[RETURN_COL] >= df[ISSUE_COL]).all(), "Return before issue detected"

    # --- late flag consistency ---
    assert LATE_COL in df.columns, f"Missing column {LATE_COL}"
    assert LATE_FLAG_COL in df.columns, f"Missing column {LATE_FLAG_COL}"
    assert df[LATE_FLAG_COL].dtype == bool, "late_flag is not boolean"

    # late_flag must be consistent with Verspätet
    assert (df[LATE_FLAG_COL] == df[LATE_COL]).all(), "late_flag differs from Verspätet"

    # --- user-session logic (only where user exists) ---
    has_user = df[USER_ID_COL].notna()

    if has_user.any():
        assert SESSION_INDEX_COL in df.columns, "session_index missing"

        # session_index must start at 1
        min_idx = (
            df.loc[has_user]
            .groupby(USER_ID_COL)[SESSION_INDEX_COL]
            .min()
        )
        assert (min_idx == 1).all(), "session_index does not start at 1 for some users"

        # session_index must be integer-like
        assert (
            df.loc[has_user, SESSION_INDEX_COL].dropna() % 1 == 0
        ).all(), "session_index is not integer-valued"

    print("[validate] all checks passed")
