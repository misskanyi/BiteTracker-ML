"""Tests for src/model_training.py — splitting, fitting, persistence, reproducibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.model_evaluation import compute_regression_metrics
from src.model_training import (
    chronological_split,
    load_model,
    prepare_features_and_target,
    save_model,
    train_decision_tree,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)


def _fake_processed_df(n_days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for school in ["A", "B"]:
        for d in dates:
            row = {col: float(rng.normal()) for col in FEATURE_COLUMNS if col not in ("month", "is_holiday", "is_exam_period", "is_school_event")}
            row.update(
                {
                    settings.DATE_COLUMN: d,
                    settings.SCHOOL_ID_COLUMN: school,
                    "month": d.month,
                    "is_holiday": 0,
                    "is_exam_period": 0,
                    "is_school_event": 0,
                    settings.TARGET_COLUMN: float(rng.normal(300, 20)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_chronological_split_no_shuffle_row_order_preserved():
    df = _fake_processed_df()
    train, test = chronological_split(df, train_ratio=0.8)
    assert train[settings.DATE_COLUMN].max() <= test[settings.DATE_COLUMN].min()


def test_chronological_split_last_20_percent_is_test():
    df = _fake_processed_df(n_days=50)
    train, test = chronological_split(df, train_ratio=0.8)
    n = len(df.sort_values(settings.DATE_COLUMN))
    assert len(test) == n - int(n * 0.8)


def test_chronological_split_no_overlap_between_periods():
    df = _fake_processed_df()
    train, test = chronological_split(df)
    train_dates = set(train[settings.DATE_COLUMN])
    test_dates = set(test[settings.DATE_COLUMN])
    assert not (train_dates & test_dates) or max(train_dates) <= min(test_dates)


def test_prepare_features_and_target_shapes_match():
    df = _fake_processed_df()
    X, y = prepare_features_and_target(df)
    assert len(X) == len(y) == len(df)
    assert list(X.columns) == FEATURE_COLUMNS


def test_prepare_features_and_target_raises_on_missing_columns():
    df = _fake_processed_df().drop(columns=["lag_1_meals"])
    with pytest.raises(ValueError):
        prepare_features_and_target(df)


@pytest.mark.parametrize("trainer", [train_linear_regression, train_decision_tree, train_random_forest, train_xgboost])
def test_models_fit_and_predict_correct_shape(trainer):
    df = _fake_processed_df()
    train, test = chronological_split(df)
    X_train, y_train = prepare_features_and_target(train)
    X_test, y_test = prepare_features_and_target(test)

    model = trainer(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (len(X_test),)
    assert not np.isnan(preds).any()


def test_saved_model_loading_round_trip(tmp_path):
    df = _fake_processed_df()
    train, _ = chronological_split(df)
    X_train, y_train = prepare_features_and_target(train)

    model = train_linear_regression(X_train, y_train)
    save_model(model, "linear_regression", models_dir=tmp_path)
    loaded = load_model("linear_regression", models_dir=tmp_path)

    np.testing.assert_allclose(model.predict(X_train), loaded.predict(X_train))


def test_metric_calculation_matches_known_values():
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 290])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["MAE"], 10.0)
    assert metrics["RMSE"] > 0
    assert metrics["R2"] <= 1.0


def test_decision_tree_reproducible_with_fixed_seed():
    df = _fake_processed_df()
    train, test = chronological_split(df)
    X_train, y_train = prepare_features_and_target(train)
    X_test, _ = prepare_features_and_target(test)

    model_a = train_decision_tree(X_train, y_train)
    model_b = train_decision_tree(X_train, y_train)

    np.testing.assert_array_equal(model_a.predict(X_test), model_b.predict(X_test))
