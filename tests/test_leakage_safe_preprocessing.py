"""Preprocessing statistics are learned from the training period only.

Imputation medians and outlier fences must come from training-period rows,
be reused unchanged for later rows, and never move information backwards in
time. Uses synthetic fixtures only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.feature_engineering import (
    FEATURE_COLUMNS,
    FOOD4EDUCATION_FEATURE_COLUMNS,
    FOOD4EDUCATION_SOURCE,
    SYNTHETIC_SOURCE,
    feature_columns_for,
)
from src.model_training import prepare_features_and_target
from src.pipeline import prepare_training_data
from src.preprocessing import (
    ALL_SCHOOLS,
    PreprocessingParams,
    clean_raw_data,
    fit_preprocessing,
    handle_missing_values,
    handle_outliers,
    run_preprocessing,
)
from src.synthetic_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    df = generate_synthetic_dataset(start_date="2023-01-01", end_date="2023-06-30")
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df


def _school_rows(values, school="A", start="2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "school_id": school, "meals_served": values})


def test_changing_the_test_period_does_not_change_training_data(raw_df):
    data = prepare_training_data(raw_df)
    future = raw_df[settings.DATE_COLUMN] > data.cutoff_date
    altered = raw_df.copy()
    altered.loc[future, "meals_served"] = altered.loc[future, "meals_served"] * 10
    altered.loc[future, "students_present"] = np.nan
    altered.loc[future, "enrolled_students"] = altered.loc[future, "enrolled_students"] * 5

    altered_data = prepare_training_data(altered)

    assert altered_data.cutoff_date == data.cutoff_date
    assert altered_data.preprocessing == data.preprocessing
    pd.testing.assert_frame_equal(altered_data.train_df, data.train_df)


def test_preprocessing_is_fitted_on_training_rows_only(raw_df):
    data = prepare_training_data(raw_df)
    cleaned = clean_raw_data(raw_df)
    expected = fit_preprocessing(cleaned[cleaned[settings.DATE_COLUMN] <= data.cutoff_date], data.cutoff_date)
    assert data.preprocessing == expected
    assert data.preprocessing.fit_end_date == str(data.cutoff_date.date())
    assert data.preprocessing.fit_rows < len(cleaned)


def test_split_has_no_shared_dates(raw_df):
    data = prepare_training_data(raw_df)
    assert data.train_df[settings.DATE_COLUMN].max() == data.cutoff_date
    assert data.test_df[settings.DATE_COLUMN].min() > data.cutoff_date


def test_test_period_outliers_are_capped_with_training_fences():
    df = _school_rows([100.0] * 10 + [100.0, 5000.0])
    params = fit_preprocessing(df.iloc[:10])
    low, high = params.outlier_bounds["meals_served"]["A"]
    result = handle_outliers(df, params)
    assert result["meals_served"].iloc[-1] == high
    assert low <= result["meals_served"].min()


def test_missing_values_are_never_back_filled():
    df = _school_rows([100.0, 100.0, 100.0])
    df["students_present"] = [np.nan, 999.0, 999.0]
    params = PreprocessingParams(
        fit_end_date=None, fit_rows=0, fill_values={"students_present": {"A": 50.0, ALL_SCHOOLS: 60.0}}
    )
    result = handle_missing_values(df, params)
    assert result["students_present"].tolist() == [50.0, 999.0, 999.0]


def test_gaps_after_a_known_value_are_forward_filled():
    df = _school_rows([100.0, 100.0, 100.0])
    df["students_present"] = [80.0, np.nan, 90.0]
    params = PreprocessingParams(fit_end_date=None, fit_rows=0, fill_values={"students_present": {"A": 50.0}})
    assert handle_missing_values(df, params)["students_present"].tolist() == [80.0, 80.0, 90.0]


def test_unknown_school_uses_the_all_schools_fallback():
    train = pd.concat([_school_rows([10.0] * 8, "A"), _school_rows([30.0] * 8, "B")])
    params = fit_preprocessing(train)
    new_school = _school_rows([10.0, 9999.0], "C")
    high = params.outlier_bounds["meals_served"][ALL_SCHOOLS][1]
    assert handle_outliers(new_school, params)["meals_served"].iloc[-1] == high


def test_params_survive_a_json_round_trip(raw_df):
    params = prepare_training_data(raw_df).preprocessing
    assert PreprocessingParams.from_dict(json.loads(json.dumps(params.to_dict()))) == params


def test_run_preprocessing_with_fit_end_date_matches_training_path(raw_df):
    data = prepare_training_data(raw_df)
    via_run = run_preprocessing(raw_df, fit_end_date=data.cutoff_date)
    pd.testing.assert_frame_equal(via_run, data.clean_df)


def test_food4education_feature_set_drops_unavailable_and_duplicate_features():
    real = feature_columns_for(FOOD4EDUCATION_SOURCE)
    assert real == FOOD4EDUCATION_FEATURE_COLUMNS
    for dropped in [
        "previous_attendance", "previous_attendance_rate", "rainfall_mm", "temperature_c",
        "is_holiday", "is_exam_period", "is_school_event", "school_term_Term 2", "school_term_Term 3",
    ]:
        assert dropped not in real
    assert set(real).issubset(FEATURE_COLUMNS)
    assert feature_columns_for(SYNTHETIC_SOURCE) == FEATURE_COLUMNS


def test_unknown_data_source_raises():
    with pytest.raises(ValueError):
        feature_columns_for("somewhere-else")


def test_prepare_features_uses_the_requested_feature_set(raw_df):
    data = prepare_training_data(raw_df)
    X, y = prepare_features_and_target(data.train_df, FOOD4EDUCATION_FEATURE_COLUMNS)
    assert list(X.columns) == FOOD4EDUCATION_FEATURE_COLUMNS
    assert len(X) == len(y)
