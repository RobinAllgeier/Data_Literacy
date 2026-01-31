# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dat"
RAW_DIR = DATA_DIR / "raw"
RAW_BORROWINGS_DIR = RAW_DIR / "borrowings"
CLOSED_DAYS_FILE = RAW_DIR / "closed_days.csv"

PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "doc" / "report"
FIGURES_DIR = REPORTS_DIR / "figures"


# ---------- Column names (ADJUST THESE) ----------

# --- core identifiers ---
AUTHOR_COL = "Autor"
TITLE_COL = "Titel"
ISBN_COL = "ISBN"
BARCODE_COL = "Barcode"
ISSUE_ID_COL = "issue_id"

# --- item metadata ---
COLLECTION_CODE_COL = "Sammlungszeichen/CCODE"
MEDIA_TYPE_COL = "Medientyp"
TOPIC_COL = "Interessenkreis"

# --- user metadata ---
USER_ID_COL = "Benutzer-Systemnummer"
USER_CATEGORY_COL = "Benutzerkategorie"

# --- dates ---
ISSUE_COL = "Ausleihdatum/Uhrzeit"
RETURN_COL = "Rückgabedatum/Uhrzeit"

# --- loan logic ---
LOAN_DURATION_COL = "Leihdauer"
DAYS_LATE_COL = "Tage_zu_spät"
LATE_COL = "Verspätet"
EXTENSIONS_COL = "Anzahl_Verlängerungen"

# --- provenance ---
SOURCE_YEAR_COL = "source_year"


# --- closed days/calender rules ---
CLOSED_DATE_COL = "schliesstag"


# ---------- Preprocessing ----------

# library open days: Tue–Sat (Mon..Sun: 0,1,1,1,1,1,0)
LIB_WEEKMASK = "0111110"

REMOVE_USER_CATEGORIES = {"MDA", "MZUZL", "SYS"}

BASE_ALLOWED_OPEN_DAYS = 28 # base allowed open days for loan duration calculation
MAX_EXTENSIONS_CAP = 6 # rule of the libary for max extensions

# ---------- Derived / feature columns ----------
LATE_FLAG_COL = "late_flag"

# user-session features
ISSUE_SESSION_COL = "issue_session"
SESSION_INDEX_COL = "session_index"
SESSION_SIZE_COL = "session_size"
SESSION_LATE_FLAG_COL = "session_late_flag"
SESSION_EXTENSION_FLAG_COL = "session_extension_flag"
EXPERIENCE_STAGE_COL = "experience_stage"
SESSION_CATEGORY_COL = "session_category"

# timing features
WEEKDAY_COL = "weekday"
HOUR_COL = "hour"

# regularity metric
USER_MODAL_WEEKDAY_COL = "user_modal_weekday"
USER_MODAL_HOUR_COL = "user_modal_hour"
USER_MATCH_TYPICAL_COL = "matches_typical_time"

# user timing features
PRECISE_HOUR_COL = "precise_hour"
USER_AVG_HOUR_COL = "user_mean_hour"
USER_STD_HOUR_COL = "user_std_hour"




# ---------- Analysis parameters ----------

# --- plot_1 learning-curve plot ---
EXPERIENCE_CUTOFF = 3          # early vs experienced threshold (session_index <= 3)
MAX_SESSION_INDEX_PLOT = 25    # cap x-axis to avoid long tail dominating
LEARNING_CURVE_SMOOTHING = 3  # moving average window size for learning curve plot

# For plot 2:

# For plot 3:

# For plot 4:
FIRST_K_THRESHOLDS = [2, 5, 10]
MIN_USER_SESSIONS = 0
MAX_SESSION_INDEX_PLOT_4 = 25
STICKINESS_CURVE_SMOOTHING = 3



@dataclass(frozen=True)
class PipelineConfig:
    """All runtime config in one place (optional, but nice)."""
    raw_input: Path
    processed_version: str = "v1"

    @property
    def processed_out_dir(self) -> Path:
        return PROCESSED_DIR / self.processed_version

    @property
    def figures_out_dir(self) -> Path:
        return FIGURES_DIR
