"""Central configuration for BiteTracker.

All paths are resolved relative to the project root so the code works
regardless of the current working directory the app/scripts are run from.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# --- Directories -----------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, UPLOADS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Dataset ----------------------------------------------------------------
# Default dataset used by the pipeline/app. Point this at a real CSV later
# without changing any other code — data_loader.py only cares about the path.
SYNTHETIC_DATA_PATH: Path = SAMPLE_DATA_DIR / "synthetic_school_meals.csv"
ACTIVE_DATA_PATH: Path = Path(os.getenv("BITETRACKER_DATA_PATH", str(SYNTHETIC_DATA_PATH)))

TARGET_COLUMN: str = "meals_served_next_day"
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
