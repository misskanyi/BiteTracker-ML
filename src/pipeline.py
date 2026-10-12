"""End-to-end data pipeline: load -> clean -> engineer features -> save.

This is the single entry point the rest of the project (model training,
web application) should call to get a model-ready dataset, so cleaning and
feature logic live in exactly one place.

`prepare_training_data` is the leakage-safe path used for model training:
it finds the chronological train/test cutoff first, fits the learned
preprocessing statistics on training-period rows only, then applies them to
every row before building features.
"""

# In simple terms: runs all the data-prep steps in order (load → clean →
# add features → save) so everything else just uses one finished file.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import settings
from src.data_loader import load_dataset
from src.feature_engineering import engineer_features
from src.model_training import chronological_split, train_cutoff_date
from src.preprocessing import PreprocessingParams, apply_preprocessing, clean_raw_data, fit_preprocessing
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingData:
    """Everything a training run needs, built without looking past the cutoff."""

    clean_df: pd.DataFrame  # cleaned school-day rows (all dates), fitted stats applied
    processed_df: pd.DataFrame  # feature-engineered rows
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    preprocessing: PreprocessingParams  # fitted on training-period rows only
    cutoff_date: pd.Timestamp  # last training date
    test_targets_capped: int  # test-period target values changed by the training-fitted fences


def prepare_training_data(
    raw_df: pd.DataFrame, train_ratio: float = settings.TRAIN_TEST_SPLIT_RATIO
) -> TrainingData:
    """Clean, fit preprocessing on the training period only, engineer features, split."""
    return build_training_data(clean_raw_data(raw_df), train_ratio)


def build_training_data(
    cleaned: pd.DataFrame, train_ratio: float = settings.TRAIN_TEST_SPLIT_RATIO
) -> TrainingData:
    """`prepare_training_data` for rows already through `preprocessing.clean_raw_data`.

    Which rows end up in training depends only on row positions and missing
    targets, never on the fitted statistics (imputation does not touch
    `meals_served`, and capping keeps missing values missing), so the cutoff
    can be found before anything is fitted. The final split is checked
    against it.
    """
    cutoff = train_cutoff_date(engineer_features(cleaned), train_ratio)

    params = fit_preprocessing(cleaned[cleaned[settings.DATE_COLUMN] <= cutoff], fit_end_date=cutoff)
    clean_df = apply_preprocessing(cleaned, params)
    processed_df = engineer_features(clean_df)
    train_df, test_df = chronological_split(processed_df, train_ratio)
    if train_df[settings.DATE_COLUMN].max() != cutoff or test_df[settings.DATE_COLUMN].min() <= cutoff:
        raise RuntimeError("Train/test split moved after preprocessing; the cutoff is no longer valid")

    test_period = cleaned[settings.DATE_COLUMN] > cutoff
    observed = cleaned.loc[test_period, "meals_served"]
    capped = int((observed.ne(clean_df.loc[test_period, "meals_served"]) & observed.notna()).sum())
    if capped:
        logger.warning(
            "%d test-period meals_served values fall outside the training-period outlier fences and were capped",
            capped,
        )
    logger.info("Preprocessing fitted on %d training-period rows (up to %s)", params.fit_rows, cutoff.date())
    return TrainingData(clean_df, processed_df, train_df, test_df, params, cutoff, capped)


def run_data_pipeline(
    raw_path: Path | None = None,
    output_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run load -> preprocess -> feature-engineer, optionally saving the result.

    Returns the processed, feature-engineered DataFrame. Preprocessing is
    fitted on the training period only (see `prepare_training_data`), so the
    saved file can be split by `run_training.py` without leakage. Pass
    `raw_path` to point at a real dataset later; everything downstream is
    unaffected as long as it matches the schema in
    `src.synthetic_data.SCHEMA_COLUMNS`.
    """
    raw_df = load_dataset(path=raw_path)
    processed_df = prepare_training_data(raw_df).processed_df

    if save:
        save_processed_dataset(processed_df, output_path)

    return processed_df


def save_processed_dataset(processed_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    """Write the feature-engineered dataset to data/processed/ (where the app reads it)."""
    out_path = output_path or (settings.PROCESSED_DATA_DIR / "school_meals_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out_path, index=False)
    logger.info("Processed dataset saved to %s (%d rows)", out_path, len(processed_df))
    return out_path


if __name__ == "__main__":
    run_data_pipeline()
