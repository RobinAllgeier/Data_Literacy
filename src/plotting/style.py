# src/plotting/style.py
from __future__ import annotations

import matplotlib.pyplot as plt
from tueplots import bundles


def apply_style() -> None:
    """
    Apply a consistent matplotlib style using tueplots.
    Call once before creating figures.
    """
    plt.rcParams.update(
        bundles.icml2024(
            column="half",   # "half" or "full"
            nrows=1,
            ncols=1,
        )
    )
