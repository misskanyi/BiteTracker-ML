"""Stages 4–7: Django pages — dashboard, historical demand, forecast & planning, forecast history,
model performance, dataset upload, data-status labels, permissions and error handling.

Most tests use `stub_artifacts` (a small synthetic dataset + fixed-weight model in a temp
directory). Tests marked with `real_artifacts` use the saved development pipeline outputs
and skip if they have not been generated.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from config import settings as ml_settings
from services import forecast_service
from services.model_service import FeatureAlignmentError, ModelLoadError, ModelMetadataError, ModelNotFoundError
from src.feature_engineering import FEATURE_COLUMNS
from web.core.errors import user_message
from web.schools.models import DatasetUpload, ForecastRun, School

pytestmark = pytest.mark.django_db
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "Str0ng-test-passw0rd!"

PAGES = ["core:dashboard", "core:history", "core:forecast", "core:forecast_history", "core:performance", "schools:upload"]


@pytest.fixture
def user(db):
    return get_user_model().objects.create_user("planner", password=PASSWORD)


@pytest.fixture
def logged_in(client, user):
    client.force_login(user)
    return client


@pytest.fixture
def uploader(client, db):
    staff = get_user_model().objects.create_user("data_officer", password=PASSWORD)
    staff.user_permissions.add(
        Permission.objects.get(codename="add_datasetupload"), Permission.objects.get(codename="view_datasetupload")
    )
    client.force_login(staff)
    return client


@pytest.fixture
def media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "uploads"
    return settings.MEDIA_ROOT


def _post_forecast(client, **data):
    payload = {"school": "SCH001", "horizon": 1, **data}
    return client.post(reverse("core:forecast"), payload)


def _no_paths(html: bytes) -> bool:
    text = html.decode()
    return "/Users/" not in text and "final_best_model.joblib" not in text and "Traceback" not in text


# --- Protection ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", PAGES)
def test_pages_require_login(client, name):
    response = client.get(reverse(name))
    assert response.status_code == 302
    assert response.url.startswith(reverse("accounts:login"))


def test_anonymous_forecast_post_is_rejected_and_not_recorded(client, stub_artifacts):
    response = _post_forecast(client)
    assert response.status_code == 302 and reverse("accounts:login") in response.url
    assert ForecastRun.objects.count() == 0


def test_forecast_post_requires_csrf_token(user, stub_artifacts):
    from django.test import Client

    csrf_client = Client(enforce_csrf_checks=True)
    csrf_client.force_login(user)
    assert _post_forecast(csrf_client).status_code == 403
    assert ForecastRun.objects.count() == 0


# --- Dashboard ------------------------------------------------------------------------------


def test_dashboard_shows_schools_model_chart_and_status(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:dashboard"))
    html = response.content.decode()
    assert response.status_code == 200
    for name in ("Greenfield Primary", "Riverside Primary", "Highlands Academy"):
        assert name in html
    assert 'data-testid="model-name">Linear Regression<' in html  # from metadata, not hardcoded
    assert 'id="dashboard-history-chart"' in html
    assert "Synthetic" in html and "SYNTHETIC DEVELOPMENT DATA" in html
    assert "No forecasts yet" in html


def test_dashboard_lists_recent_forecast_runs(logged_in, stub_artifacts):
    _post_forecast(logged_in, horizon=7)
    html = logged_in.get(reverse("core:dashboard")).content.decode()
    assert "Next 7 school days" in html
    assert "No forecasts yet" not in html


def test_dashboard_without_data_or_model_degrades_gracefully(logged_in, stub_artifacts):
    stub_artifacts.processed_path.unlink()
    stub_artifacts.metadata_path.unlink()
    response = logged_in.get(reverse("core:dashboard"))
    assert response.status_code == 200
    assert b'data-testid="data-error"' in response.content
    assert b"Historical meal-demand data has not been generated yet" in response.content
    assert _no_paths(response.content)


# --- Historical demand ------------------------------------------------------------------------------


def test_history_default_shows_first_school_table_and_chart(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"))
    html = response.content.decode()
    assert response.status_code == 200
    assert 'data-testid="history-table"' in html
    assert 'id="history-chart"' in html
    assert "Greenfield Primary" in html  # first school by name


def test_history_filters_by_school_and_date(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"), {"school": "SCH002", "start": "2023-03-01", "end": "2023-03-10"})
    html = response.content.decode()
    assert response.status_code == 200
    assert "Riverside Primary (SCH002)" in html
    dates = re.findall(r'<td class="text-nowrap">(\d+ \w+ 2023)</td>', html)
    assert dates and all(" Mar 2023" in d for d in dates)
    assert response.context["records"] == len(dates)


def test_history_rejects_reversed_date_range(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"), {"school": "SCH001", "start": "2023-03-10", "end": "2023-03-01"})
    assert b"start date must be on or before the end date" in response.content


def test_history_rejects_unknown_school(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"), {"school": "SCH999"})
    assert response.status_code == 200
    assert b"Select a valid choice" in response.content


def test_history_empty_range_shows_empty_state(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"), {"school": "SCH001", "start": "2030-01-01", "end": "2030-02-01"})
    assert b'data-testid="history-empty"' in response.content


def test_history_paginates(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:history"), {"school": "SCH001", "start": "2023-01-01", "end": "2023-04-30", "page": 2})
    assert response.status_code == 200
    assert response.context["page"].number == 2


# --- Forecast & planning ------------------------------------------------------------------------------


def test_forecast_form_offers_exact_school_day_horizons(logged_in, stub_artifacts):
    html = logged_in.get(reverse("core:forecast")).content.decode()
    for label in ("Next school day", "Next 7 school days", "Next 30 school days"):
        assert label in html
    assert "calendar days" not in html
    assert "not from today" in html
    assert json.loads(re.search(r'id="school-defaults" type="application/json">(.*?)</script>', html).group(1))["SCH001"]


def test_forecast_form_prefills_from_query(logged_in, stub_artifacts):
    form = logged_in.get(reverse("core:forecast"), {"school": "SCH003", "horizon": "30"}).context["form"]
    assert form.initial == {"school": "SCH003", "horizon": "30"}


@pytest.mark.parametrize("horizon", [1, 7, 30])
def test_forecast_each_horizon_creates_run_and_shows_result(logged_in, stub_artifacts, user, horizon):
    response = _post_forecast(logged_in, horizon=horizon, expected_rainfall_mm="4.5")
    run = ForecastRun.objects.get()
    assert response.status_code == 302 and response.url == reverse("core:forecast_run", args=[run.pk])

    assert run.user == user and run.horizon_days == horizon and run.school_code == "SCH001"
    assert run.model_label == "Linear Regression"
    assert run.results_label == "SYNTHETIC DEVELOPMENT RESULTS"
    assert run.assumptions == {"expected_rainfall_mm": 4.5}
    assert len(run.result_summary["days"]) == horizon
    assert run.day1_recommended_meals >= run.day1_predicted_meals
    assert run.origin_date < run.first_forecast_date <= run.last_forecast_date

    html = logged_in.get(response.url).content.decode()
    assert "Predicted demand" in html and "Recommended preparation" in html
    assert "not a statistical confidence interval" in html
    assert f"Forecast generated from available development data ending {run.origin_date.day} {run.origin_date:%B %Y}" in html
    assert html.count('<td class="text-nowrap">') >= horizon
    assert 'id="forecast-chart"' in html


def test_recorded_forecast_matches_service_result_exactly(logged_in, stub_artifacts):
    """Persistence parity: what is stored and shown equals the service output for the same inputs."""
    _post_forecast(logged_in, horizon=7, expected_attendance="300", expected_temperature_c="22")
    run = ForecastRun.objects.get()
    direct = forecast_service.generate_forecast("SCH001", 7, expected_attendance=300, expected_temperature_c=22)
    stored = run.result_summary["days"]
    assert [d["predicted_meals_served"] for d in stored] == [d["predicted_meals_served"] for d in direct.days]
    assert [d["recommended_meals"] for d in stored] == [d["recommended_meals"] for d in direct.days]
    assert run.total_predicted_meals == direct.totals["predicted_meals"]


def test_forecast_with_capacity_shows_allocation(logged_in, stub_artifacts):
    response = _post_forecast(logged_in, available_capacity="5")
    html = logged_in.get(response.url).content.decode()
    assert 'data-testid="allocation"' in html
    assert "exceeds planned capacity" in html or "exceeds forecast demand" in html or "matches" in html


def test_forecast_links_to_imported_school(logged_in, stub_artifacts):
    School.objects.create(school_id="SCH001", name="Greenfield Primary")
    _post_forecast(logged_in)
    assert ForecastRun.objects.get().school.school_id == "SCH001"


@pytest.mark.parametrize(
    "data, message",
    [
        ({"horizon": 5}, "Select a valid choice"),
        ({"horizon": "abc"}, "Select a valid choice"),
        ({"school": "SCH999"}, "Select a valid choice"),
        ({"expected_rainfall_mm": "-3"}, "greater than or equal to 0"),
        ({"expected_temperature_c": "80"}, "less than or equal to 50"),
        ({"expected_attendance": "many"}, "Enter a number"),
        ({"available_capacity": "-1"}, "greater than or equal to 0"),
    ],
)
def test_forecast_invalid_input_shows_message_and_records_nothing(logged_in, stub_artifacts, data, message):
    response = _post_forecast(logged_in, **data)
    assert response.status_code == 200
    assert message in response.content.decode()
    assert ForecastRun.objects.count() == 0


def test_forecast_missing_model_shows_friendly_error(logged_in, stub_artifacts):
    stub_artifacts.model_path.unlink()
    response = _post_forecast(logged_in)
    assert response.status_code == 200
    assert b"The forecasting model has not been generated yet" in response.content
    assert _no_paths(response.content)
    assert ForecastRun.objects.count() == 0


def test_forecast_feature_mismatch_shows_friendly_error(logged_in, stub_artifacts, make_model):
    stub_artifacts.write_model(make_model(FEATURE_COLUMNS[:-1]))
    response = _post_forecast(logged_in)
    assert b"does not match the features BiteTracker builds" in response.content
    assert _no_paths(response.content)
    assert ForecastRun.objects.count() == 0


def test_forecast_insufficient_history_shows_message(logged_in, stub_artifacts):
    frame = stub_artifacts.frame
    stub_artifacts.write_processed(pd.concat([frame[frame["school_id"] != "SCH001"], frame[frame["school_id"] == "SCH001"].head(5)]))
    response = _post_forecast(logged_in)
    assert b"at least 14 are needed" in response.content
    assert ForecastRun.objects.count() == 0


def test_forecast_page_without_data_shows_friendly_error(logged_in, stub_artifacts):
    stub_artifacts.processed_path.unlink()
    response = logged_in.get(reverse("core:forecast"))
    assert b'data-testid="data-error"' in response.content
    assert _no_paths(response.content)


def test_forecast_never_trains_during_request(logged_in, stub_artifacts, monkeypatch):
    from sklearn.linear_model import LinearRegression

    monkeypatch.setattr(LinearRegression, "fit", lambda *a, **k: pytest.fail("model was trained during a request"))
    assert _post_forecast(logged_in, horizon=30).status_code == 302


def test_forecast_run_missing_returns_404(logged_in, stub_artifacts):
    response = logged_in.get(reverse("core:forecast_run", args=[999]))
    assert response.status_code == 404
    assert b'data-testid="error-404"' in response.content or response.status_code == 404


def test_forecast_history_lists_and_filters_runs(logged_in, stub_artifacts):
    _post_forecast(logged_in, school="SCH001")
    _post_forecast(logged_in, school="SCH002", horizon=7)
    html = logged_in.get(reverse("core:forecast_history")).content.decode()
    assert "Greenfield Primary" in html and "Riverside Primary" in html
    filtered = logged_in.get(reverse("core:forecast_history"), {"school": "SCH002"}).content.decode()
    assert "Riverside Primary" in filtered
    assert len(re.findall(r'href="/forecast/runs/\d+/"', filtered)) == 1


def test_real_model_forecast_through_django(logged_in, real_artifacts):
    """End-to-end with the saved development Random Forest (or whichever model the metadata names)."""
    for horizon in (1, 7, 30):
        response = _post_forecast(logged_in, horizon=horizon)
        assert response.status_code == 302
    runs = ForecastRun.objects.all()
    assert {r.horizon_days for r in runs} == {1, 7, 30}
    assert {r.model_label for r in runs} == {real_artifacts.metadata["selected_model"]}
    html = logged_in.get(reverse("core:forecast_run", args=[runs[0].pk])).content.decode()
    assert "Tuned XGBoost" not in html or "XGBoost" in real_artifacts.metadata["selected_model"]


# --- Model performance ------------------------------------------------------------------------------


def test_performance_page_shows_saved_metrics(logged_in, stub_artifacts):
    html = logged_in.get(reverse("core:performance")).content.decode()
    assert "SYNTHETIC DEVELOPMENT RESULTS" in html
    assert 'data-testid="selected-model">Linear Regression' in html
    assert 'data-testid="validation-mae">11.00<' in html
    assert 'data-testid="test-mae">10.00<' in html and 'data-testid="test-rmse">12.00<' in html
    assert 'data-testid="test-r2">0.900<' in html
    assert "Persistence / Last Observation" in html and "7-Day Historical Mean" in html
    assert "lowest <strong>mean validation MAE</strong>" in html
    assert "not an accuracy percentage" in html
    assert 'id="baselines-chart"' in html


def test_real_performance_page_never_calls_r2_accuracy(logged_in, real_artifacts):
    html = logged_in.get(reverse("core:performance")).content.decode()
    r2 = real_artifacts.metadata["final_test_r2"]
    assert f"{r2:.3f}" in html
    assert f"{r2 * 100:.1f}%" not in html
    assert "accuracy" not in html.replace("not an accuracy percentage", "")
    assert "unequivocally the best model" in html  # selection caveat shown when folds vary


def test_performance_missing_artifacts_shows_friendly_error(logged_in, stub_artifacts):
    stub_artifacts.naive_metrics_path.unlink()
    response = logged_in.get(reverse("core:performance"))
    assert b'data-testid="performance-error"' in response.content
    assert _no_paths(response.content)


# --- Data upload ------------------------------------------------------------------------------------


def _csv_upload(frame: pd.DataFrame, name: str = "school_data.csv") -> SimpleUploadedFile:
    return SimpleUploadedFile(name, frame.to_csv(index=False).encode(), content_type="text/csv")


def _valid_upload_frame(stub_artifacts) -> pd.DataFrame:
    return stub_artifacts.frame[["date", "school_id", "school_name", "meals_served", "students_present", "enrolled_students"]].head(80)


def test_upload_requires_permission(logged_in, stub_artifacts):
    response = logged_in.get(reverse("schools:upload"))
    assert response.status_code == 403
    assert b'data-testid="error-403"' in response.content
    assert logged_in.post(reverse("schools:upload"), {"file": _csv_upload(_valid_upload_frame(stub_artifacts))}).status_code == 403
    assert DatasetUpload.objects.count() == 0


def test_upload_page_states_no_replacement_or_retraining(uploader, stub_artifacts):
    html = uploader.get(reverse("schools:upload")).content.decode()
    assert "does <strong>not</strong> replace the active dataset" in html
    assert "does <strong>not</strong> retrain the model" in html


def test_valid_upload_is_validated_recorded_and_stored(uploader, stub_artifacts, media_root):
    model_mtime = stub_artifacts.model_path.stat().st_mtime_ns
    data_mtime = stub_artifacts.processed_path.stat().st_mtime_ns

    response = uploader.post(reverse("schools:upload"), {"file": _csv_upload(_valid_upload_frame(stub_artifacts))})
    record = DatasetUpload.objects.get()
    assert response.status_code == 302 and response.url == reverse("schools:upload_detail", args=[record.pk])
    assert record.validation_status == "valid"
    assert record.uploaded_by.username == "data_officer"
    assert record.n_records == 80 and record.forecast_readiness == "READY"
    assert record.processing_status == "not_processed"
    assert record.file and Path(record.file.path).is_relative_to(media_root)
    assert "date" in record.validation_summary["columns"]

    # Active dataset and model untouched, nothing forecast.
    assert stub_artifacts.model_path.stat().st_mtime_ns == model_mtime
    assert stub_artifacts.processed_path.stat().st_mtime_ns == data_mtime
    assert ForecastRun.objects.count() == 0

    html = uploader.get(response.url).content.decode()
    assert 'data-testid="validation-status"' in html and "Valid." in html
    assert "The active dataset and model are unchanged" in html


def test_upload_missing_required_columns_is_invalid_and_not_stored(uploader, stub_artifacts, media_root):
    frame = _valid_upload_frame(stub_artifacts).drop(columns=["meals_served"])
    uploader.post(reverse("schools:upload"), {"file": _csv_upload(frame)})
    record = DatasetUpload.objects.get()
    assert record.validation_status == "invalid"
    assert not record.file
    assert any("meals_served" in e for e in record.validation_summary["errors"])


def test_upload_corrupt_excel_is_unreadable(uploader, stub_artifacts, media_root):
    bad = SimpleUploadedFile("broken.xlsx", b"not really an excel file", content_type="application/vnd.ms-excel")
    response = uploader.post(reverse("schools:upload"), {"file": bad})
    record = DatasetUpload.objects.get()
    assert record.validation_status == "unreadable" and not record.file
    html = uploader.get(response.url).content.decode()
    assert "could not be read" in html
    assert "Traceback" not in html


def test_upload_real_excel_file_validates(uploader, stub_artifacts, media_root):
    buffer = io.BytesIO()
    _valid_upload_frame(stub_artifacts).to_excel(buffer, index=False)
    upload = SimpleUploadedFile("school_data.xlsx", buffer.getvalue())
    uploader.post(reverse("schools:upload"), {"file": upload})
    assert DatasetUpload.objects.get().validation_status == "valid"


@pytest.mark.parametrize("name", ["data.txt", "data.xls", "data.exe", "noextension"])
def test_upload_rejects_unsupported_file_types(uploader, stub_artifacts, media_root, name):
    response = uploader.post(reverse("schools:upload"), {"file": SimpleUploadedFile(name, b"a,b\n1,2\n")})
    assert response.status_code == 200
    assert b"Unsupported file type" in response.content
    assert DatasetUpload.objects.count() == 0


def test_upload_rejects_oversized_file(uploader, stub_artifacts, media_root, settings):
    settings.MAX_UPLOAD_SIZE_BYTES = 1024 * 1024
    big = SimpleUploadedFile("big.csv", b"x" * (1024 * 1024 + 1))
    response = uploader.post(reverse("schools:upload"), {"file": big})
    assert b"larger than the 1 MB limit" in response.content
    assert DatasetUpload.objects.count() == 0


def test_upload_without_file_shows_error(uploader, stub_artifacts):
    response = uploader.post(reverse("schools:upload"), {})
    assert b"This field is required" in response.content


# --- Data-status labels --------------------------------------------------------------------------------


def test_banner_is_synthetic_for_synthetic_artifacts(logged_in, stub_artifacts):
    html = logged_in.get(reverse("core:dashboard")).content.decode()
    assert "data-banner--synthetic" in html and "SYNTHETIC DEVELOPMENT DATA" in html


def test_banner_shows_real_label_only_for_real_data(logged_in, stub_artifacts, make_metadata, monkeypatch, tmp_path):
    monkeypatch.setattr(ml_settings, "ACTIVE_DATA_PATH", tmp_path / "food4education.csv")
    stub_artifacts.write_metadata(make_metadata(results_label="FOOD4EDUCATION RESEARCH RESULTS"))
    html = logged_in.get(reverse("core:dashboard")).content.decode()
    assert "data-banner--real" in html and "FOOD4EDUCATION RESEARCH RESULTS" in html
    assert "SYNTHETIC DEVELOPMENT DATA" not in html


def test_banner_unknown_when_metadata_missing_for_non_synthetic_data(logged_in, stub_artifacts, monkeypatch, tmp_path):
    monkeypatch.setattr(ml_settings, "ACTIVE_DATA_PATH", tmp_path / "food4education.csv")
    stub_artifacts.metadata_path.unlink()
    html = logged_in.get(reverse("core:dashboard")).content.decode()
    assert "data-banner--unknown" in html and "DATA STATUS UNKNOWN" in html


# --- Error messages and production settings ----------------------------------------------------------------


@pytest.mark.parametrize("error", [ModelNotFoundError, ModelLoadError, ModelMetadataError, FeatureAlignmentError])
def test_artifact_error_messages_hide_paths(error):
    message = user_message(error("No trained model found at /Users/someone/project/models/final_best_model.joblib"))
    assert "/Users" not in message and ".joblib" not in message


def test_custom_404_page(logged_in):
    response = logged_in.get("/no-such-page/")
    assert response.status_code == 404


def _settings_subprocess(**env_overrides) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DJANGO_")}
    env.update({"DJANGO_SETTINGS_MODULE": "bitetracker_web.settings", **env_overrides})
    code = "import django; django.setup(); from django.conf import settings; print(settings.DEBUG, settings.SESSION_COOKIE_SECURE, settings.ALLOWED_HOSTS)"
    return subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)


def test_production_settings_require_allowed_hosts():
    result = _settings_subprocess(DJANGO_DEBUG="False", DJANGO_SECRET_KEY="x" * 50, DJANGO_ALLOWED_HOSTS="")
    assert result.returncode != 0 and "DJANGO_ALLOWED_HOSTS must be set" in result.stderr


def test_production_settings_enable_secure_cookies():
    result = _settings_subprocess(DJANGO_DEBUG="False", DJANGO_SECRET_KEY="x" * 50, DJANGO_ALLOWED_HOSTS="bitetracker.example.org")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False True ['bitetracker.example.org']"
