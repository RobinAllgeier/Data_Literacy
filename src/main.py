# src/main.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    RAW_BORROWINGS_DIR,
    CLOSED_DAYS_FILE,
    PROCESSED_DIR,
    PipelineConfig,
)
from src.io import (
    load_borrowings_raw,
    load_closed_days,
    save_processed,
    load_processed_version
)
from src.preprocess import preprocess_borrowings
from src.features import add_features
from src.validate import validate_borrowings

from src.plotting.plot_1_libary_visit_clock import make_plot as plot1
from src.plotting.plot_1_libary_visit_clock import print_user_statistics as user_stats
from src.plotting.plot_2_learning_curve import make_plot as plot2
from src.plotting.plot_3_overview import make_plot as plot3
from src.plotting.plot_4_stickiness_to_media_type import make_plot as plot4
from src.plotting.plot_4_stickiness_to_media_type import print_media_type_session_statistics as print_media_type_stats



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borrowings pipeline: raw -> processed + features")
    p.add_argument(
        "--version",
        default="v1",
        help="processed dataset version folder (e.g. v1, v2)"
    )
    p.add_argument(
        "--use-processed",
        action="store_true",
        help="load newest processed dataset and skip preprocessing & feature generation"
    )
    return p.parse_args()





def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(raw_input=RAW_BORROWINGS_DIR, processed_version=args.version)

    # --------------------------------------------------
    # FAST PATH: load processed data only
    # --------------------------------------------------
    if args.use_processed:
        df_feat = load_processed_version(PROCESSED_DIR, args.version)

    # --------------------------------------------------
    # FULL PIPELINE
    # --------------------------------------------------
    else:
        # 1) load raw
        df_raw = load_borrowings_raw(cfg.raw_input)
        closed = load_closed_days(CLOSED_DAYS_FILE)

        # 2) preprocess
        df_clean = preprocess_borrowings(df_raw, closed_days=closed)

        # 3) features
        df_feat = add_features(df_clean)

        # 4) validate
        validate_borrowings(df_feat)

        # 5) save processed
        save_processed(
            df_feat,
            cfg.processed_out_dir,
            version=cfg.processed_version
        )

        print(f"[main] saved processed dataset to: {cfg.processed_out_dir}")

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------
    user_stats(df_feat)
    print_media_type_stats(df_feat)
    cfg.figures_out_dir.mkdir(parents=True, exist_ok=True)
    plot1(df_feat, cfg.figures_out_dir / "plot_1_clock_plot.pdf")
    plot2(df_feat, cfg.figures_out_dir / "plot_2_learning_curve.pdf")
    plot3(df_feat, cfg.figures_out_dir / "plot_3_overview.pdf")
    plot4(df_feat, cfg.figures_out_dir / "plot_4_media_type_stickiness.pdf")


if __name__ == "__main__":
    main()



