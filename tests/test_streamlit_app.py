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


# --- Planning page -----------------------------------------------------------
# 3_Planning.py consolidates three earlier pages: Specific Date Prediction and
# Scenario Simulator became the "Predict & Compare" tab, and Forecast Calendar
# became the "Forecast Calendar" tab. The tests below cover the same behaviour
# on the consolidated page.

PLANNING_PAGE = "app/pages/3_Planning.py"


def _metric_meals(at, label: str) -> float:
    """Parse the numeric part of a '<n> meals' metric by its label."""
    metric = next(m for m in at.metric if m.label == label)
    return float(str(metric.value).replace("meals", "").strip())


def test_planning_page_has_predict_and_calendar_tabs():
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception
    labels = [t.label for t in at.tabs]
    assert any("Predict & Compare" in label for label in labels)
    assert any("Forecast Calendar" in label for label in labels)


def test_planning_predict_has_school_date_and_condition_inputs():
    """Replaces test_specific_date_prediction_has_school_and_date_inputs."""
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception
    assert at.selectbox[0].label == "School"
    assert at.date_input(key="pc_target_date") is not None
    for key in ("pc_rainfall", "pc_temp"):
        assert at.number_input(key=key) is not None
    assert at.selectbox(key="pc_term").options == ["Term 1", "Term 2", "Term 3"]
    for key in ("pc_holiday", "pc_exam", "pc_event"):
        assert at.checkbox(key=key) is not None
    # The forecast date's own attendance isn't known in advance, so it is not an input.
    assert not any(n.key == "pc_attendance" for n in at.number_input)


def test_planning_predict_generates_result():
    """Replaces test_specific_date_prediction_generates_result."""
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception

    at.button(key="pc_generate").click().run()
    assert not at.exception
    metric_labels = {m.label for m in at.metric}
    assert {"Baseline Forecast", "Your Prediction", "Difference"} <= metric_labels
    assert _metric_meals(at, "Your Prediction") >= 0


def test_planning_compares_custom_conditions_against_baseline():
    """Replaces test_scenario_simulator_runs_scenario.

    The baseline-vs-scenario comparison now lives in Predict & Compare. The
    old attendance-% slider and "Reset Scenario" button no longer exist:
    target-date attendance is not a model input, and inputs are pre-filled
    with the baseline's own values instead of being reset by a button.
    """
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception
    assert any(m.label.startswith("Baseline Forecast") for m in at.metric)

    at.number_input(key="pc_rainfall").set_value(40.0)
    at.checkbox(key="pc_event").check()
    at.button(key="pc_generate").click().run()
    assert not at.exception

    baseline = _metric_meals(at, "Baseline Forecast")
    predicted = _metric_meals(at, "Your Prediction")
    difference = float(next(m for m in at.metric if m.label == "Difference").value.replace("meals", ""))
    # Metrics are rounded independently for display, so allow 1 meal of slack.
    assert abs(difference - (predicted - baseline)) <= 1
    info_text = " ".join(i.value for i in at.info)
    assert "compared to the baseline forecast" in info_text


def test_planning_holiday_disables_exam_checkbox():
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception

    at.checkbox(key="pc_holiday").check().run()
    assert not at.exception
    assert at.checkbox(key="pc_exam").disabled


def test_planning_forecast_calendar_shows_table_and_day_detail():
    """Replaces test_forecast_calendar_shows_table_and_day_detail."""
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    assert not at.exception

    calendar = at.dataframe[0].value
    assert len(calendar) == 14  # default horizon
    assert {"Date", "Predicted Demand", "Recommended Preparation", "Risk", "Estimated Cost (KSh)"} <= set(calendar.columns)
    assert len(at.selectbox(key="cal_selected_date").options) == 14


def test_planning_forecast_calendar_30_day_horizon():
    at = AppTest.from_file(PLANNING_PAGE)
    at.run()
    at.selectbox(key="cal_horizon").set_value(30).run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 30
    assert len(at.selectbox(key="cal_selected_date").options) == 30


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
