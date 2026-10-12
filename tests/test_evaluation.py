"""Tests for src/model_evaluation.py — metrics and chronological splitting.

What this file checks:
- MAE and RMSE are correct for a perfect prediction and for known values.
- The split is chronological (train before test), roughly 80/20, with no shuffling and
  no shared rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from src.model_evaluation import chronological_train_test_split, compute_regression_metrics


def test_compute_regression_metrics_perfect_prediction():
    y_true = pd.Series([10, 20, 30, 40])
    y_pred = pd.Series([10, 20, 30, 40])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0
    assert metrics["RMSE"] == 0
    assert metrics["R2"] == 1.0


def test_compute_regression_metrics_known_values():
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([12, 18, 33])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], (2 + 2 + 3) / 3)
    assert metrics["RMSE"] > 0
    assert metrics["R2"] <= 1.0


def _dated_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({settings.DATE_COLUMN: dates, "value": range(n)})


def test_chronological_split_keeps_train_before_test():
    df = _dated_df()
    train, test = chronological_train_test_split(df)
    assert train[settings.DATE_COLUMN].max() < test[settings.DATE_COLUMN].min()


def test_chronological_split_ratio_approximately_80_20():
    df = _dated_df(100)
    train, test = chronological_train_test_split(df, train_ratio=0.8)
    assert 0.75 <= len(train) / len(df) <= 0.85


def test_chronological_split_no_shuffling_no_row_overlap():
    df = _dated_df()
    train, test = chronological_train_test_split(df)
    overlap = set(train.index) & set(test.index)
    assert len(overlap) == 0
    assert len(train) + len(test) == len(df)
