import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def log_pearson_spearman(df: pd.DataFrame, x_col: str, y_col: str):
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)

    n = int(mask.sum())
    print(f"{x_col} vs {y_col} (n={n})")

    if n < 3:
        print("Pearson: not enough data")
        print("Spearman: not enough data")
        return

    r, p = pearsonr(x[mask], y[mask])
    rho, p_s = spearmanr(x[mask], y[mask])

    print(f"Pearson  r   = {r:.4f}   p-value (\"null hypothesis: no correlation\") = {p:.3g}")
    print(f"Spearman rho = {rho:.4f}   p-value (\"null hypothesis: no correlation\") = = {p_s:.3g}")
