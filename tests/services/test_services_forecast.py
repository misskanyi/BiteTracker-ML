"""Stage 2: forecast_service — parity with src.forecasting, horizons, planning output, errors."""

from __future__ import annotations

import json
from datetime import timedelta

import joblib
import pandas as pd
import pytest

from config import settings
from services import ServiceError, data_service, forecast_service
from services.data_service import DataNotFoundError, UnknownSchoolError
from services.forecast_service import (
    HORIZONS,
    InsufficientHistoryError,
    InvalidForecastRequestError,
    generate_forecast,
)
from services.model_service import ModelNotFoundError
from src.budgeting import estimate_meal_cost
from src.decision_support import build_decision_support_result
from src.forecasting import forecast_school_demand
from src.resource_allocation import compute_allocation

INPUT_CASES = {
    "recent-average assumptions": {},
    "user-provided inputs": {"expected_attendance": 350.0, "expected_rainfall_mm": 10.5, "expected_temperature_c": 24.0},
}


def _direct_forecast(processed_path, model_path, school_id, horizon, inputs):
    """The existing path, with no service layer: read the CSV, load the model, call src.forecasting."""
    df = pd.read_csv(processed_path)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return forecast_school_demand(df, school_id, horizon, model=joblib.load(model_path), **inputs)


# --- PARITY: the service layer must not change forecasting behaviour ----------------


@pytest.mark.parametrize("inputs", INPUT_CASES.values(), ids=INPUT_CASES.keys())
@pytest.mark.parametrize("horizon", sorted(HORIZONS))
@pytest.mark.parametrize("school_id", ["SCH001", "SCH002", "SCH003"])
def test_parity_service_matches_direct_src_forecasting_real_model(real_artifacts, school_id, horizon, inputs):
    """Same school, dataset, horizon, inputs and saved model → identical predictions."""
    direct = _direct_forecast(real_artifacts.processed_path, real_artifacts.model_path, school_id, horizon, inputs)
    result = generate_forecast(school_id, horizon, **inputs)

    pd.testing.assert_frame_equal(result.forecast_frame, direct, check_exact=True)
    assert [d["predicted_meals_served"] for d in result.days] == direct["predicted_meals_served"].tolist()
    assert [d["date"] for d in result.days] == direct["date"].tolist()


@pytest.mark.parametrize("inputs", INPUT_CASES.values(), ids=INPUT_CASES.keys())
@pytest.mark.parametrize("horizon", sorted(HORIZONS))
def test_parity_service_matches_direct_src_forecasting_stub_model(stub_artifacts, horizon, inputs):
    """Same check on the stub artifacts, so parity is tested even without pipeline outputs."""
    direct = _direct_forecast(stub_artifacts.processed_path, stub_artifacts.model_path, "SCH002", horizon, inputs)
    result = generate_forecast("SCH002", horizon, **inputs)
    pd.testing.assert_frame_equal(result.forecast_frame, direct, check_exact=True)


def test_day1_decision_support_matches_existing_forecast_page_logic(stub_artifacts):
    """Day-1 summary = build_decision_support_result with the former Streamlit Forecast page's inputs."""
    result = generate_forecast("SCH001", 7)
    history = data_service.get_school_history("SCH001")
    expected = build_decision_support_result(
        predicted_meals=result.days[0]["predicted_meals_served"],
        recent_std=float(history["meals_served"].tail(14).std() or 0.0),
        horizon_days=1,
        horizon_label="Next school day",
        model_label="Linear Regression",
    )
    for key in ("predicted_meals", "recommended_meals", "buffer_meals", "buffer_pct", "risk_level", "uncertainty"):
        assert result.decision_support[key] == getattr(expected, key)


def test_planning_rows_match_existing_calendar_logic(stub_artifacts):
    """Each day's recommendation uses that day's distance ahead, as the former Streamlit Planning calendar did."""
    result = generate_forecast("SCH003", 30)
    history = data_service.get_school_history("SCH003")
    recent_std = float(history["meals_served"].tail(14).std() or 0.0)
    for day in result.days:
        expected = build_decision_support_result(
            predicted_meals=day["predicted_meals_served"], recent_std=recent_std,
            horizon_days=day["day_number"], horizon_label="", model_label="x",
        )
        assert day["recommended_meals"] == expected.recommended_meals
        assert day["risk_level"] == expected.risk_level
        assert day["recommended_meals"] >= day["predicted_meals"]


# --- Horizons and forecast origin ---------------------------------------------------


def test_horizon_labels_are_school_days():
    assert HORIZONS == {1: "Next school day", 7: "Next 7 school days", 30: "Next 30 school days"}
    assert not any("calendar" in label.lower() for label in HORIZONS.values())


@pytest.mark.parametrize("horizon", [1, 7, 30])
def test_each_horizon_returns_that_many_consecutive_school_days(stub_artifacts, horizon):
    result = generate_forecast("SCH001", horizon)
    dates = [d["date"] for d in result.days]
    assert len(dates) == horizon
    assert result.horizon_label == HORIZONS[horizon]
    assert all(d.weekday() < 5 for d in dates)
    assert dates[0] > result.origin_date
    assert result.first_forecast_date == dates[0] and result.last_forecast_date == dates[-1]
    # No school day is skipped between the origin and the last forecast day.
    expected, current = [], result.origin_date
    while len(expected) < horizon:
        current += timedelta(days=1)
        if current.weekday() < 5:
            expected.append(current)
    assert dates == expected


def test_forecast_origin_is_last_recorded_date_not_today(stub_artifacts):
    result = generate_forecast("SCH001", 1)
    last = stub_artifacts.frame[settings.DATE_COLUMN].max().date()
    assert result.origin_date == last
    assert result.origin_note == f"Forecast generated from available development data ending {last.day} {last:%B %Y}."


def test_real_forecast_origin_note_names_end_of_development_data(real_artifacts):
    result = generate_forecast("SCH001", 1)
    assert result.origin_date == data_service.get_latest_date()
    assert result.origin_note.startswith("Forecast generated from available development data ending ")
    assert result.results_label == "SYNTHETIC DEVELOPMENT RESULTS"


def test_result_is_plain_python_and_json_serialisable(stub_artifacts):
    result = generate_forecast("SCH001", 7, available_capacity=300)
    data = result.to_dict()
    json.dumps(data)
    assert data["origin_date"] == result.origin_date.isoformat()
    assert len(data["forecast_frame"]) == 7


def test_cost_and_capacity_use_existing_budgeting_and_allocation(stub_artifacts):
    result = generate_forecast("SCH001", 1, available_capacity=10)
    day1 = result.days[0]["predicted_meals_served"]
    cost_per_meal = float(data_service.get_school_history("SCH001")["average_cost_per_meal"].tail(14).mean())
    assert result.cost_estimate == estimate_meal_cost(day1, cost_per_meal)
    assert result.days[0]["estimated_cost"] == estimate_meal_cost(day1, cost_per_meal)["daily_cost"]
    assert result.allocation == compute_allocation(day1, 10)
    assert result.allocation["available_capacity"] == 10
    assert result.allocation["gap"] == round(day1 - 10)
    assert result.allocation["status"] in {"Shortage", "Surplus", "Balanced"}


# --- Model label comes from metadata --------------------------------------------------


def test_forecast_model_label_comes_from_metadata(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(selected_model="Linear Regression"))
    result = generate_forecast("SCH001", 1)
    assert result.model_name == "Linear Regression"
    assert result.decision_support["model_label"] == "Linear Regression"


def test_real_forecast_never_reports_tuned_xgboost_when_metadata_says_otherwise(real_artifacts):
    selected = real_artifacts.metadata["selected_model"]
    result = generate_forecast("SCH001", 1)
    assert result.decision_support["model_label"] == selected
    if "XGBoost" not in selected:
        assert "XGBoost" not in result.decision_support["model_label"]


# --- Error handling ---------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [0, 2, 14, 31, -7, "7", None, True, 7.5])
def test_invalid_horizon_rejected(stub_artifacts, horizon):
    with pytest.raises(InvalidForecastRequestError, match="Forecast horizon must be one of"):
        generate_forecast("SCH001", horizon)


@pytest.mark.parametrize(
    "inputs, message",
    [
        ({"expected_rainfall_mm": -1}, "rainfall.*at least 0"),
        ({"expected_attendance": -5}, "attendance.*at least 0"),
        ({"expected_temperature_c": 80}, "temperature.*at most 50"),
        ({"expected_temperature_c": -40}, "temperature.*at least -10"),
        ({"expected_rainfall_mm": float("nan")}, "finite"),
        ({"expected_attendance": float("inf")}, "finite"),
        ({"expected_rainfall_mm": "heavy"}, "must be a number"),
        ({"available_capacity": -1}, "capacity.*at least 0"),
    ],
)
def test_invalid_inputs_rejected(stub_artifacts, inputs, message):
    with pytest.raises(InvalidForecastRequestError, match=f"(?i){message}"):
        generate_forecast("SCH001", 1, **inputs)


def test_unknown_school_rejected(stub_artifacts):
    with pytest.raises(UnknownSchoolError, match="SCH999"):
        generate_forecast("SCH999", 1)


def test_insufficient_history_rejected(stub_artifacts):
    frame = stub_artifacts.frame
    short = frame[frame[settings.SCHOOL_ID_COLUMN] == "SCH001"].head(10)
    stub_artifacts.write_processed(pd.concat([frame[frame[settings.SCHOOL_ID_COLUMN] != "SCH001"], short]))
    with pytest.raises(InsufficientHistoryError, match="10 recorded school day.*at least 14"):
        generate_forecast("SCH001", 1)
    assert generate_forecast("SCH002", 1).days  # other schools unaffected


def test_missing_processed_data_rejected(stub_artifacts):
    stub_artifacts.processed_path.unlink()
    with pytest.raises(DataNotFoundError, match="scripts/train_models.py"):
        generate_forecast("SCH001", 1)


def test_missing_model_rejected_through_forecast(stub_artifacts):
    stub_artifacts.model_path.unlink()
    with pytest.raises(ModelNotFoundError):
        generate_forecast("SCH001", 1)


def test_forecast_errors_are_service_errors():
    for error in (InvalidForecastRequestError, InsufficientHistoryError, UnknownSchoolError, DataNotFoundError):
        assert issubclass(error, ServiceError)
