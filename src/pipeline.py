"""End-to-end data pipeline: load -> clean -> engineer features -> save.

This is the single entry point the rest of the project (model training,
Streamlit app) should call to get a model-ready dataset, so cleaning and
feature logic live in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.preprocessing import run_preprocessing
from src.utils import get_logger

logger = get_logger(__name__)


def run_data_pipeline(
    raw_path: Path | None = None,
    output_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run load -> preprocess -> feature-engineer, optionally saving the result.

    Returns the processed, feature-engineered DataFrame. Pass `raw_path` to
    point at a real dataset later; everything downstream is unaffected as
    long as it matches the schema in `src.synthetic_data.SCHEMA_COLUMNS`.
    """
    raw_df = load_dataset(path=raw_path)
    clean_df = run_preprocessing(raw_df)
    processed_df = engineer_features(clean_df)

    if save:
        out_path = output_path or (settings.PROCESSED_DATA_DIR / "school_meals_processed.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(out_path, index=False)
        logger.info("Processed dataset saved to %s (%d rows)", out_path, len(processed_df))

    return processed_df


if __name__ == "__main__":
    run_data_pipeline()
