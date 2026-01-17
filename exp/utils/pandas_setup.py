import pandas as pd


def setup_pandas():
    """Standard pandas display settings for analysis notebooks."""

    # Show up to 200 columns when displaying DataFrames (avoid column truncation)
    pd.set_option("display.max_columns", 200)

    # Set the total display width in characters for DataFrame output
    # Helps keep tables readable in wide notebooks
    pd.set_option("display.width", 140)

    # Format floating-point numbers consistently:
    # - thousands separator
    # - 4 decimal places
    # Improves readability and avoids scientific notation in most cases
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
