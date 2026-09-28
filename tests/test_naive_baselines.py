"""Tests for src/naive_baselines.py — per-school, strictly-past naive forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.feature_engineering import engineer_features
from src.model_training import chronological_split
from src.naive_baselines import (
    HISTORICAL_MEAN_7,
    PERSISTENCE,
    PREDICTION_COLUMNS,
    compare_to_naive_baselines,
    compute_naive_forecasts,
    evaluate_naive_baselines,
    naive_forecasts_for_period,
)
from src.preprocessing import run_preprocessing
from src.synthetic_data import generate_synthetic_dataset

PERSIST = PREDICTION_COLUMNS[PERSISTENCE]
MEAN7 = PREDICTION_COLUMNS[HISTORICAL_MEAN_7]


def _history():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    a = pd.DataFrame({"school_id": "A", "date": dates, "meals_served": np.arange(1, 11) * 10.0})
    b = pd.DataFrame({"school_id": "B", "date": dates, "meals_served": np.arange(1, 11) * 1000.0})
    # Interleave rows by date, as the real processed data is.
    return pd.concat([a, b]).sort_values(["date", "school_id"]).reset_index(drop=True)


def test_persistence_is_previous_observation_of_same_school():
    out = compute_naive_forecasts(_history()).set_index(["school_id", "date"])
    day5 = pd.Timestamp("2024-01-05")
    assert out.loc[("A", day5), PERSIST] == 40.0
    assert out.loc[("B", day5), PERSIST] == 4000.0
    assert np.isnan(out.loc[("A", pd.Timestamp("2024-01-01")), PERSIST])


def test_seven_day_mean_excludes_target_day_and_needs_seven_obs():
    out = compute_naive_forecasts(_history()).set_index(["school_id", "date"])
    assert np.isnan(out.loc[("A", pd.Timestamp("2024-01-07")), MEAN7])  # only 6 prior obs
    # 2024-01-08: prior 7 days are 10..70 -> mean 40; the day's own 80 is excluded.
    assert out.loc[("A", pd.Timestamp("2024-01-08")), MEAN7] == pytest.approx(40.0)
    assert out.loc[("B", pd.Timestamp("2024-01-10")), MEAN7] == pytest.approx(np.mean(np.arange(3, 10) * 1000.0))


def test_future_values_do_not_change_past_forecasts():
    history = _history()
    before = compute_naive_forecasts(history)
    altered = history.copy()
    altered.loc[altered["date"] >= "2024-01-08", "meals_served"] = 1e9
    after = compute_naive_forecasts(altered)
    mask = before["date"] <= "2024-01-08"
    pd.testing.assert_frame_equal(before.loc[mask, [PERSIST, MEAN7]], after.loc[mask, [PERSIST, MEAN7]])


def test_missing_observations_are_skipped():
    history = _history()
    history.loc[(history["school_id"] == "A") & (history["date"] == "2024-01-04"), "meals_served"] = np.nan
    out = compute_naive_forecasts(history).set_index(["school_id", "date"])
    assert out.loc[("A", pd.Timestamp("2024-01-05")), PERSIST] == 30.0


def test_duplicate_school_dates_rejected():
    history = pd.concat([_history(), _history().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        compute_naive_forecasts(history)


def test_forecasts_for_period_require_enough_history():
    history = _history()
    with pytest.raises(ValueError, match="prior history"):
        naive_forecasts_for_period(history, history[history["date"] <= "2024-01-03"])


def test_on_pipeline_data_persistence_matches_lag_1_feature():
    raw = generate_synthetic_dataset(start_date="2023-01-01", end_date="2023-06-30")
    clean = run_preprocessing(raw)
    processed = engineer_features(clean)
    _, test_df = chronological_split(processed)
    forecasts = naive_forecasts_for_period(clean, test_df)
    np.testing.assert_allclose(forecasts[PERSIST].values, test_df["lag_1_meals"].values)
    np.testing.assert_allclose(forecasts[MEAN7].values, test_df["rolling_mean_7"].values)

    metrics, predictions = evaluate_naive_baselines(clean, test_df)
    assert list(metrics["Method"]) == [PERSISTENCE, HISTORICAL_MEAN_7]
    assert set(metrics["Type"]) == {"Naive baseline"}
    assert len(predictions) == len(test_df)
    assert (metrics["Test MAE"] > 0).all()
    np.testing.assert_allclose(predictions["actual_meals"].values, test_df[settings.TARGET_COLUMN].values)


def test_compare_to_naive_baselines_reports_losses_honestly():
    naive = pd.DataFrame(
        [
            {"Method": PERSISTENCE, "Test MAE": 20.0, "Test RMSE": 25.0},
            {"Method": HISTORICAL_MEAN_7, "Test MAE": 10.0, "Test RMSE": 12.0},
        ]
    )
    out = compare_to_naive_baselines("XGBoost", {"MAE": 15.0, "RMSE": 18.0}, naive).set_index("Naive Baseline")
    assert out.loc[PERSISTENCE, "MAE Improvement"] == 5.0
    assert out.loc[PERSISTENCE, "MAE Improvement (%)"] == pytest.approx(25.0)
    assert bool(out.loc[PERSISTENCE, "Model Beats Baseline (MAE)"])
    assert out.loc[HISTORICAL_MEAN_7, "MAE Improvement"] == -5.0
    assert not bool(out.loc[HISTORICAL_MEAN_7, "Model Beats Baseline (MAE)"])
