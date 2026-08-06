"""Evaluation metrics and chronological splitting utilities.

Kept separate from model_training.py (Phase 3) so both training and tests
can depend on a single, simple definition of "how do we score a model" and
"how do we split time-ordered data without leaking the future".
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import settings


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return MAE, RMSE, and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def classify_generalization_gap(train_mae: float, test_mae: float, train_r2: float, test_r2: float) -> str:
    """Label a model's train/test gap as Overfitting / Moderate Gap / Good Generalization.

    Thresholds: test MAE >50% higher than train MAE, or a train-test R² gap
    >0.15, flags overfitting. A gap under 10% MAE / 0.05 R² is labelled good
    generalization. Anything between is a moderate, worth-watching gap.
    These are heuristics for a small synthetic dataset, not universal rules.
    """
    mae_gap_pct = (test_mae - train_mae) / max(train_mae, 1e-9) * 100
    r2_gap = train_r2 - test_r2
    if mae_gap_pct > 50 or r2_gap > 0.15:
        return "Overfitting"
    if mae_gap_pct < 10 and abs(r2_gap) < 0.05:
        return "Good Generalization"
    return "Moderate Gap"


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = settings.TRAIN_TEST_SPLIT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-sorted DataFrame into train/test by time, not randomly.

    Splits on the overall date range (not per-school row count) so every
    school's test rows come from the same trailing time window — this
    mirrors how the model will actually be used (forecasting a future
    period for all schools at once) and avoids leaking a later date's
    pattern into training via one school's row ordering.
    """
    df = df.sort_values(settings.DATE_COLUMN)
    unique_dates = sorted(df[settings.DATE_COLUMN].unique())
    cutoff_idx = max(0, int(len(unique_dates) * train_ratio) - 1)
    cutoff_date = unique_dates[cutoff_idx]
    train = df[df[settings.DATE_COLUMN] <= cutoff_date]
    test = df[df[settings.DATE_COLUMN] > cutoff_date]
    return train, test
