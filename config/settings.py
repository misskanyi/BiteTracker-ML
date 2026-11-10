"""Central configuration for BiteTracker.

All paths are resolved relative to the project root so the code works
regardless of the current working directory the app/scripts are run from.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# The local .env (credentials, live data URLs) is not loaded under pytest, so tests never depend
# on, or act on, a developer's secrets. BITETRACKER_SKIP_DOTENV=1 does the same elsewhere.
if "pytest" not in sys.modules and not os.getenv("BITETRACKER_SKIP_DOTENV"):
    load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Directories -----------------------------------------------------------
# Writable state (trained model, model metadata, processed historical CSV,
# metrics, figures, user uploads and, in bitetracker_web/settings.py, the
# SQLite database) lives under STATE_DIR. Unset (local development), STATE_DIR
# is the project root, so every path is exactly where it always was. In
# production set BITETRACKER_STATE_DIR (e.g. /var/lib/bitetracker) so state
# survives redeploys of the code directory.
_state_dir = os.getenv("BITETRACKER_STATE_DIR", "").strip()
STATE_DIR: Path = Path(_state_dir).expanduser().resolve() if _state_dir else PROJECT_ROOT
STATE_DIR_IS_EXTERNAL: bool = bool(_state_dir)

DATA_DIR: Path = STATE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
# The committed synthetic sample is code, not state: it always stays in the project.
SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "data" / "sample"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
MODELS_DIR: Path = STATE_DIR / "models"
REPORTS_DIR: Path = STATE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, UPLOADS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Dataset ----------------------------------------------------------------
# Default dataset used by the pipeline/app. Point this at a real CSV later
# without changing any other code — data_loader.py only cares about the path.
SYNTHETIC_DATA_PATH: Path = SAMPLE_DATA_DIR / "synthetic_school_meals.csv"
ACTIVE_DATA_PATH: Path = Path(os.getenv("BITETRACKER_DATA_PATH", str(SYNTHETIC_DATA_PATH)))

# Real Food4Education data, read live from its Google Sheets share link (see
# src/google_sheet_loader.py). When set, it replaces ACTIVE_DATA_PATH as the
# default dataset. Keep the link in .env, not in committed code.
GOOGLE_SHEET_URL: str = os.getenv("BITETRACKER_GOOGLE_SHEET_URL", "").strip()
USE_GOOGLE_SHEET: bool = bool(GOOGLE_SHEET_URL)
GOOGLE_SHEET_DATASET_NAME: str = "Food4Education Google Sheet"
# The sheet has no meal-cost column; budgeting uses this value (KSh) when it
# is set. Unset means unknown, so the cost column is left empty, not zero.
_cost_per_meal = os.getenv("BITETRACKER_COST_PER_MEAL_KSH", "").strip()
COST_PER_MEAL_KSH: float | None = float(_cost_per_meal) if _cost_per_meal else None
# Planning assumption for meal budgets in the web app (services/planning_service.py).
# Kept separate from COST_PER_MEAL_KSH so it never changes the dataset; it is a
# user-adjustable assumption, not verified Food4Education expenditure.
PLANNING_COST_PER_MEAL_KES: str = os.getenv("BITETRACKER_PLANNING_COST_PER_MEAL_KES", "27").strip() or "27"

# Meals served on the target date; each processed row is that target date
# (see src/feature_engineering.py).
TARGET_COLUMN: str = "target_meals_served"
DATE_COLUMN: str = "date"
SCHOOL_ID_COLUMN: str = "school_id"

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# --- Train/test split ---------------------------------------------------------
TRAIN_TEST_SPLIT_RATIO: float = 0.8  # chronological 80/20 split
CV_N_SPLITS: int = 5  # TimeSeriesSplit folds

# --- Synthetic data generation parameters ------------------------------------
SYNTHETIC_SCHOOLS = [
    {"school_id": "SCH001", "school_name": "Greenfield Primary", "county": "Nakuru", "sub_county": "Naivasha", "base_enrollment": 420},
    {"school_id": "SCH002", "school_name": "Riverside Primary", "county": "Kisumu", "sub_county": "Nyando", "base_enrollment": 610},
    {"school_id": "SCH003", "school_name": "Highlands Academy", "county": "Kiambu", "sub_county": "Limuru", "base_enrollment": 340},
]

# Two academic years of school-day records (Kenyan calendar approximated).
SYNTHETIC_START_DATE: str = "2023-01-01"
SYNTHETIC_END_DATE: str = "2024-12-20"

# --- Postgres (prepared for future use, not used in v1) ----------------------
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "bitetracker")
DB_USER: str = os.getenv("DB_USER", "bitetracker_user")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
