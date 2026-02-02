from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tueplots.constants.color import rgb
import numpy as np
import pandas as pd

from src.config import ISSUE_COL, EXTENSIONS_COL, PROCESSED_DIR
from src.plotting.style import apply_style


def make_plot(df: pd.DataFrame, outpath) -> None:
    apply_style()
    outpath = Path(outpath) if outpath is not None else None
    t0 = time.perf_counter()
    
    # Load pre-cleaning data to calculate removed records
    pre_cleaning_file = PROCESSED_DIR / "borrowings_2019_2025.csv"
    df_pre_clean = pd.read_csv(pre_cleaning_file, sep=";", encoding="utf-8")
    df_pre_clean[ISSUE_COL] = pd.to_datetime(df_pre_clean[ISSUE_COL], errors='coerce')
    df_pre_clean['year'] = df_pre_clean[ISSUE_COL].dt.year
    pre_clean_counts = df_pre_clean.groupby('year').size()
    
    df = df.copy()
    df[ISSUE_COL] = pd.to_datetime(df[ISSUE_COL], errors='coerce')
    df['year'] = df[ISSUE_COL].dt.year
    
    years = sorted(df['year'].dropna().unique())
    
    # Count cleaned borrowings per year
    cleaned_counts = df.groupby('year').size()
    
    # Calculate removed counts per year
    removed_counts = pd.Series([
        pre_clean_counts.get(year, 0) - cleaned_counts.get(year, 0)
        for year in years
    ], index=years)
    
    # # Count number of items that were extended (at least once)
    # if EXTENSIONS_COL in df.columns:
    #     extensions_count = df.groupby('year')[EXTENSIONS_COL].apply(lambda x: (x > 0).sum())
    # else:
    #     extensions_count = pd.Series(0, index=years)
    
    # # Count late loans
    # if 'late_bool' in df.columns:
    #     late_count = df.groupby('year')['late_bool'].sum()
    # elif 'Verspätet' in df.columns:
    #     late_count = df.groupby('year')['Verspätet'].sum()
    # else:
    #     late_count = pd.Series(0, index=years)
    
    # Ensure all series have the same index
    cleaned_counts = cleaned_counts.reindex(years, fill_value=0)
    removed_counts = removed_counts.reindex(years, fill_value=0)
    # extensions_count = extensions_count.reindex(years, fill_value=0)
    # late_count = late_count.reindex(years, fill_value=0)
    
    fig, ax1 = plt.subplots()
    x = np.arange(len(years))
    width = 0.6
    
    # Left axis: Total borrowings as bars
    bars1 = ax1.bar(x, cleaned_counts.values, width,
                    label='Number of Borrowings', color=rgb.tue_blue, alpha=0.85, zorder=2)
    
    # # Extensions as thinner bars inside (indicator)
    # bars_ext = ax1.bar(x, extensions_count.values, width * 0.5,
    #                    label='Extensions (count)', color=rgb.tue_gold, alpha=0.9, zorder=3)
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Borrowings')
    ax1.set_xticks(x)
    ax1.set_xticklabels([int(y) for y in years])
    
    # Format y-axis in thousands
    ax1.set_ylim(200000, None)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    ax1.grid(axis='y', alpha=0.3, linewidth=0.8, color='0.88', zorder=0)
    ax1.set_axisbelow(True)
    
    # Right axis: Rates in %
    ax2 = ax1.twinx()
    
    # Calculate percentages
    # late_rate = (late_count / cleaned_counts * 100).fillna(0)
    removed_rate = (removed_counts / (cleaned_counts + removed_counts) * 100).fillna(0)
    
    # Plot lines for rates
    # line_late = ax2.plot(x, late_rate.values, linewidth=1.3, marker='o', markersize=2.5,
    #                      color=rgb.tue_red, label='Late Return Rate', zorder=5)
    line_removed = ax2.plot(x,
                            removed_rate.values,
                            linewidth=1.3,
                            marker='o',
                            markersize=2.5,
                            color=rgb.pn_orange,
                            label='Removed Data Rate',
                            zorder=5)
    
    ax2.set_ylabel('Rate of removed data')
    # ax2.set_ylim(0, max(late_rate.max(), removed_rate.max()) * 1.15)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}\%'))
    
    ax1.set_title('Annual Library Borrowing Volume and Portion of Removed Data')
    fig.tight_layout()
    
    # Save
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    print(f"[plot3] total time: {time.perf_counter() - t0:.2f}s")

