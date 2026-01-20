# src/main.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    RAW_BORROWINGS_DIR,
    CLOSED_DAYS_FILE,
    PipelineConfig,
)
from src.io import load_borrowings_raw, load_closed_days, save_processed
from src.preprocess import preprocess_borrowings
from src.features import add_features
from src.validate import validate_borrowings

from src.plotting.plot_1_learning_curve import make_plot as plot1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borrowings pipeline: raw -> processed + features")
    p.add_argument("--version", default="v1", help="processed dataset version folder (e.g. v1, v2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(raw_input=RAW_BORROWINGS_DIR, processed_version=args.version)

    # 1) load
    df_raw = load_borrowings_raw(cfg.raw_input)
    closed = load_closed_days(CLOSED_DAYS_FILE)

    # 2) preprocess
    df_clean = preprocess_borrowings(df_raw, closed_days=closed)

    # 3) features
    df_feat = add_features(df_clean)

    # 4) validate
    validate_borrowings(df_feat)

    # 5) save processed
    save_processed(df_feat, cfg.processed_out_dir, version=cfg.processed_version)

    print(f"[main] saved processed dataset to: {cfg.processed_out_dir}")

    cfg.figures_out_dir.mkdir(parents=True, exist_ok=True)

    plot1(df_feat, cfg.figures_out_dir / "plot_1_learning_curve.pdf")


if __name__ == "__main__":
    main()
