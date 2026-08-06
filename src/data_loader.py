"""Load and schema-validate the school-meals dataset.

Kept deliberately dumb: it reads whatever CSV `config.settings.ACTIVE_DATA_PATH`
points at (synthetic by default) and checks it matches the expected schema.
Swapping the synthetic CSV for a real dataset later requires no code changes
here as long as the column names match.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings
from src.synthetic_data import SCHEMA_COLUMNS, generate_synthetic_dataset, save_synthetic_dataset
from src.utils import get_logger

logger = get_logger(__name__)


class SchemaValidationError(ValueError):
    """Raised when a dataset does not match the expected BiteTracker schema."""


def validate_schema(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Raise `SchemaValidationError` if any required column is missing."""
    required_columns = required_columns or SCHEMA_COLUMNS
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Dataset is missing required columns: {missing}")


def load_dataset(path: Path | None = None, generate_if_missing: bool = True) -> pd.DataFrame:
    """Load the active dataset (synthetic by default), validating its schema.

    If `path` points at the default synthetic dataset and the file does not
    exist yet, it is generated automatically (helps first-time setup).
    """
    data_path = path or settings.ACTIVE_DATA_PATH

    if not data_path.exists():
        if generate_if_missing and data_path == settings.SYNTHETIC_DATA_PATH:
            logger.warning("Synthetic dataset not found at %s — generating it now.", data_path)
            save_synthetic_dataset(generate_synthetic_dataset(), data_path)
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path)
    validate_schema(df)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df
