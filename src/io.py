# src/io.py
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import SOURCE_YEAR_COL
from src.config import CLOSED_DATE_COL


def load_borrowings_raw(borrowings_dir: Path) -> pd.DataFrame:
    """
    Load all borrowings_*.csv files from a directory and concatenate them.
    Adds SOURCE_YEAR_COL extracted from filename (e.g. borrowings_2021.csv).
    """
    if not borrowings_dir.exists() or not borrowings_dir.is_dir():
        raise FileNotFoundError(f"Borrowings directory not found: {borrowings_dir}")

    files = sorted(borrowings_dir.glob("borrowings_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files matching 'borrowings_*.csv' in: {borrowings_dir}"
        )

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f, sep=";", engine="python")

        match = re.search(r"(19|20)\d{2}", f.name)
        if match:
            df[SOURCE_YEAR_COL] = int(match.group())

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)




def load_closed_days(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Closed days file not found: {path}")

    df = pd.read_csv(path, sep=";", quotechar='"', encoding="utf-8")

    if CLOSED_DATE_COL not in df.columns:
        raise KeyError(f"Expected column '{CLOSED_DATE_COL}' in {path.name}")

    return df



def save_processed(df: pd.DataFrame, out_dir: Path, version: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "borrowings.parquet", index=False)

    metadata = {
        "version": version,
        "rows": int(len(df)),
        "created_at": datetime.utcnow().isoformat(),
        "columns": list(df.columns),
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_processed_version(processed_root: Path, version: str) -> pd.DataFrame:
    """
    Load a specific processed dataset version, e.g. version='v1'.
    """
    out_dir = processed_root / version

    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError(f"Processed version not found: {out_dir}")

    parquet_path = out_dir / "borrowings.parquet"
    meta_path = out_dir / "metadata.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing borrowings.parquet in {out_dir}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {out_dir}")

    print(f"[io] loading processed dataset version: {version}")
    return pd.read_parquet(parquet_path)
