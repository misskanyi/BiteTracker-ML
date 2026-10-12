"""Stage 2: data_service (CSV-backed history) and performance_service (saved metrics)."""

from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd
import pytest

import src.forecasting
from config import settings
from services import data_service, performance_service
from services.data_service import DataLoadError, DataNotFoundError, UnknownSchoolError
from services.model_service import ModelMetadataError
from services.performance_service import PerformanceDataError

# --- data_service -------------------------------------------------------------------------


def test_list_schools_returns_school_metadata(stub_artifacts):
    schools = data_service.list_schools()
    assert [s["school_id"] for s in schools] == ["SCH001", "SCH003", "SCH002"]  # sorted by name
    first = schools[0]
    assert first["school_name"] == "Greenfield Primary"
    assert {"county", "sub_county", "enrolled_students"} <= set(first)
    assert isinstance(first["enrolled_students"], int)


def test_school_history_is_chronological_and_filterable(stub_artifacts):
    history = data_service.get_school_history("SCH002")
    assert set(history[settings.SCHOOL_ID_COLUMN]) == {"SCH002"}
    assert history[settings.DATE_COLUMN].is_monotonic_increasing

    window = data_service.get_school_history("SCH002", start=date(2023, 3, 1), end=date(2023, 3, 31))
    assert not window.empty
    assert window[settings.DATE_COLUMN].min().date() >= date(2023, 3, 1)
    assert window[settings.DATE_COLUMN].max().date() <= date(2023, 3, 31)


def test_unknown_school_history_raises(stub_artifacts):
    with pytest.raises(UnknownSchoolError):
        data_service.get_school_history("NOPE")


def test_date_range_and_latest_date(stub_artifacts):
    frame = stub_artifacts.frame
    expected_end = frame[settings.DATE_COLUMN].max().date()
    assert data_service.get_date_range() == {"start": frame[settings.DATE_COLUMN].min().date(), "end": expected_end}
    assert data_service.get_latest_date() == expected_end
    assert data_service.get_latest_date("SCH001") <= expected_end


def test_processed_data_cached_and_returned_as_copy(stub_artifacts, monkeypatch):
    reads = []
    real_read_csv = pd.read_csv
    monkeypatch.setattr(data_service.pd, "read_csv", lambda *a, **k: reads.append(a) or real_read_csv(*a, **k))

    first = data_service.load_processed_data()
    first.loc[:, "meals_served"] = -1  # mutating a returned frame must not touch the cache
    second = data_service.load_processed_data()
    assert len(reads) == 1
    assert (second["meals_served"] >= 0).all()


def test_missing_processed_data_raises(stub_artifacts):
    stub_artifacts.processed_path.unlink()
    with pytest.raises(DataNotFoundError):
        data_service.list_schools()


def test_processed_data_without_required_columns_raises(stub_artifacts):
    stub_artifacts.write_processed(stub_artifacts.frame.drop(columns=["meals_served"]))
    with pytest.raises(DataLoadError, match="meals_served"):
        data_service.load_processed_data()


def test_real_data_service_reads_development_csv(real_artifacts):
    csv = pd.read_csv(real_artifacts.processed_path, parse_dates=[settings.DATE_COLUMN])
    assert sorted(data_service.school_ids()) == sorted(csv[settings.SCHOOL_ID_COLUMN].unique())
    assert data_service.get_latest_date() == csv[settings.DATE_COLUMN].max().date()


# --- performance_service -----------------------------------------------------------------------


def test_performance_reads_metadata_and_baselines(stub_artifacts):
    perf = performance_service.get_model_performance()
    assert perf["results_label"] == "SYNTHETIC DEVELOPMENT RESULTS"
    assert perf["selected_model"] == "Linear Regression"
    assert perf["validation"]["mae"] == 11.0
    assert perf["test"] == {"mae": 10.0, "rmse": 12.0, "r2": 0.9, "evaluation": None}
    assert [b["name"] for b in perf["naive_baselines"]] == ["Persistence / Last Observation", "7-Day Historical Mean"]
    assert [c["baseline"] for c in perf["baseline_comparison"]] == ["Persistence / Last Observation", "7-Day Historical Mean"]


def test_performance_does_not_load_the_model(stub_artifacts, monkeypatch):
    monkeypatch.setattr(src.forecasting.joblib, "load", lambda *a, **k: pytest.fail("model was loaded"))
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("model was loaded"))
    performance_service.get_model_performance()


def test_performance_missing_naive_metrics_raises(stub_artifacts):
    stub_artifacts.naive_metrics_path.unlink()
    with pytest.raises(PerformanceDataError, match="scripts/train_models.py"):
        performance_service.get_model_performance()


def test_performance_missing_baseline_row_raises(stub_artifacts):
    df = pd.read_csv(stub_artifacts.naive_metrics_path)
    df[df["Method"] != "7-Day Historical Mean"].to_csv(stub_artifacts.naive_metrics_path, index=False)
    with pytest.raises(PerformanceDataError, match="7-Day Historical Mean"):
        performance_service.get_model_performance()


def test_performance_missing_baseline_comparison_raises(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(naive_baseline_comparison=[]))
    with pytest.raises(PerformanceDataError, match="baseline comparison"):
        performance_service.get_model_performance()


def test_performance_missing_metadata_raises(stub_artifacts):
    stub_artifacts.metadata_path.unlink()
    with pytest.raises(ModelMetadataError):
        performance_service.get_model_performance()


def test_real_performance_matches_saved_artifacts_exactly(real_artifacts):
    """Every reported number is the one the pipeline saved — nothing recomputed or invented."""
    meta = real_artifacts.metadata
    naive = pd.read_csv(real_artifacts.naive_metrics_path).set_index("Method")
    perf = performance_service.get_model_performance()

    assert perf["results_label"] == "SYNTHETIC DEVELOPMENT RESULTS"
    assert perf["selected_model"] == meta["selected_model"]
    assert perf["selection"]["criterion"] == meta["selection_criterion"]
    assert perf["validation"]["mae"] == meta["validation_mae"]
    assert perf["validation"]["rmse"] == meta["validation_rmse"]
    assert perf["test"]["mae"] == meta["final_test_mae"]
    assert perf["test"]["rmse"] == meta["final_test_rmse"]
    assert perf["test"]["r2"] == meta["final_test_r2"]
    for baseline in perf["naive_baselines"]:
        assert baseline["test_mae"] == naive.loc[baseline["name"], "Test MAE"]
        assert baseline["test_rmse"] == naive.loc[baseline["name"], "Test RMSE"]
        assert baseline["test_r2"] == naive.loc[baseline["name"], "Test R2"]
    saved = {row["Naive Baseline"]: row for row in meta["naive_baseline_comparison"]}
    for row in perf["baseline_comparison"]:
        assert row["mae_improvement"] == saved[row["baseline"]]["MAE Improvement"]
        assert row["model_beats_baseline"] == saved[row["baseline"]]["Model Beats Baseline (MAE)"]
    json.dumps(perf)
