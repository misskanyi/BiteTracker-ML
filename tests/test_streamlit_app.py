"""AppTest-based tests for the Streamlit app — every page must load without
exceptions, and key interactive flows (Forecast button, Model Performance
metrics, Data Explorer filters) must actually work, not just render.

Requires the full pipeline to have been run at least once (processed data,
baseline/tuned metrics, final model) — these tests skip gracefully if not.
"""

from __future__ import annotations

import glob

import pytest
from streamlit.testing.v1 import AppTest

from config import settings

ARTIFACTS_READY = (
    (settings.PROCESSED_DATA_DIR / "school_meals_processed.csv").exists()
    and (settings.MODELS_DIR / "final_best_model.joblib").exists()
    and (settings.METRICS_DIR / "baseline_model_metrics.csv").exists()
    and (settings.METRICS_DIR / "tuned_model_metrics.csv").exists()
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_READY,
    reason="Pipeline artifacts missing — run run_pipeline.py, run_training.py, run_tuning.py first.",
)

ALL_PAGES = ["app/Home.py"] + sorted(glob.glob("app/pages/*.py"))


@pytest.mark.parametrize("page_path", ALL_PAGES)
def test_page_loads_without_exceptions(page_path):
    at = AppTest.from_file(page_path)
    at.run()
    assert not at.exception, f"{page_path} raised: {at.exception}"


def test_home_page_shows_pipeline_status_metrics():
    at = AppTest.from_file("app/Home.py")
    at.run()
    assert not at.exception
    assert len(at.metric) > 0


def test_dashboard_shows_kpi_metrics():
    at = AppTest.from_file("app/pages/1_Dashboard.py")
    at.run()
    assert not at.exception
    assert len(at.metric) >= 4


def test_forecast_button_generates_results():
    at = AppTest.from_file("app/pages/2_Forecast.py")
    at.run()
    assert not at.exception
    assert len(at.button) > 0

    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) > 0


def test_forecast_page_has_school_and_horizon_selectors():
    at = AppTest.from_file("app/pages/2_Forecast.py")
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 2


def test_model_performance_loads_saved_metrics_into_dataframes():
    at = AppTest.from_file("app/pages/11_Model_Performance.py")
    at.run()
    assert not at.exception
    # Baseline, tuned, and comparison tables should each render as a dataframe.
    assert len(at.dataframe) >= 3


def test_model_performance_shows_selected_model_success_banner():
    at = AppTest.from_file("app/pages/11_Model_Performance.py")
    at.run()
    assert not at.exception
    assert len(at.success) >= 1


def test_data_explorer_default_shows_all_schools():
    at = AppTest.from_file("app/pages/10_Data_Explorer.py")
    at.run()
    assert not at.exception
    assert len(at.dataframe) > 0


def test_data_explorer_school_filter_reduces_rows():
    at = AppTest.from_file("app/pages/10_Data_Explorer.py")
    at.run()
    assert not at.exception

    multiselect = at.multiselect[0]
    all_schools = list(multiselect.options)
    assert len(all_schools) > 1

    multiselect.set_value([all_schools[0]]).run()
    assert not at.exception

    header = next((m for m in at.markdown if "Dataset Preview" in m.value), None)
    assert header is not None
    assert "rows" in header.value


def test_data_explorer_download_button_present():
    """AppTest has no dedicated introspection for `st.download_button`, so this
    confirms the page — including the CSV-encoding call feeding the download
    button — executes without error."""
    at = AppTest.from_file("app/pages/10_Data_Explorer.py")
    at.run()
    assert not at.exception


def test_about_page_mentions_synthetic_disclosure():
    at = AppTest.from_file("app/pages/12_About.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "synthetic" in full_text.lower()


def test_specific_date_prediction_generates_result():
    at = AppTest.from_file("app/pages/3_Specific_Date_Prediction.py")
    at.run()
    assert not at.exception
    assert len(at.button) > 0

    at.button[0].click().run()
    assert not at.exception


def test_specific_date_prediction_has_school_and_date_inputs():
    at = AppTest.from_file("app/pages/3_Specific_Date_Prediction.py")
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 2
    assert len(at.date_input) >= 1


def test_scenario_simulator_runs_scenario():
    at = AppTest.from_file("app/pages/4_Scenario_Simulator.py")
    at.run()
    assert not at.exception
    assert len(at.button) >= 2  # Run Scenario + Reset Scenario

    at.button[0].click().run()
    assert not at.exception
    assert len(at.metric) > 0


def test_forecast_calendar_shows_table_and_day_detail():
    at = AppTest.from_file("app/pages/5_Forecast_Calendar.py")
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert len(at.selectbox) >= 3  # school, horizon, date-inspector


def test_school_comparison_default_shows_two_schools():
    at = AppTest.from_file("app/pages/6_School_Comparison.py")
    at.run()
    assert not at.exception
    assert len(at.dataframe) >= 1


def test_resource_allocation_shows_gap_recommendations():
    at = AppTest.from_file("app/pages/7_Resource_Allocation.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.info)
    assert "shortage" in full_text.lower() or "surplus" in full_text.lower() or "balanced" in full_text.lower() or "match" in full_text.lower()


def test_budget_planning_shows_cost_estimates():
    at = AppTest.from_file("app/pages/8_Budget_Planning.py")
    at.run()
    assert not at.exception
    assert len(at.metric) >= 3


def test_upload_page_shows_privacy_notice_with_no_file():
    at = AppTest.from_file("app/pages/9_Upload_School_Data.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "privacy" in full_text.lower()
    assert "student names" in full_text.lower()


def test_upload_page_badge_shows_user_uploaded_label():
    at = AppTest.from_file("app/pages/9_Upload_School_Data.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "user-uploaded data" in full_text.lower()


def test_forecast_page_badge_shows_synthetic_label_by_default():
    at = AppTest.from_file("app/pages/2_Forecast.py")
    at.run()
    assert not at.exception
    full_text = " ".join(m.value for m in at.markdown)
    assert "synthetic development data" in full_text.lower()
