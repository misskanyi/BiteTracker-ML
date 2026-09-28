"""BiteTracker — Upload School Data: validate and process a school-provided dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import pandas as pd
import streamlit as st
from theme import configure_page, page_header, quality_score_badge

from data_access import save_uploaded_dataset
from src.data_validation import (
    UnsupportedFileTypeError,
    check_feature_compatibility,
    load_uploaded_file,
    validate_upload,
)
from src.preprocessing import run_preprocessing
from src.feature_engineering import engineer_features
from src.data_loader import SchemaValidationError

configure_page("Upload School Data", icon="📤")
page_header(
    "Upload School Data",
    subtitle="Bring your own historical meal-service records for validation and forecasting.",
    icon="📤",
    badge_text="User-Uploaded Data",
)

st.markdown(
    """
<div class="bite-card">
<b>Privacy notice:</b> upload only aggregated, school-level daily records. Do <b>not</b> include
student names, admission numbers, phone numbers, addresses, or any other personally-identifying
information. Required columns: <code>date</code>, <code>school_id</code>, <code>school_name</code>,
<code>meals_served</code>. Everything else is optional but improves forecast quality. Uploaded files
are stored locally under <code>data/uploads/</code> and are never committed to the project's Git history.
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("No file uploaded yet. Select a CSV or Excel file above to begin.")
    st.stop()

try:
    raw_df = load_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
except UnsupportedFileTypeError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # malformed file content
    st.error(f"Could not read this file: {exc}")
    st.stop()

st.markdown("### Preview")
st.dataframe(raw_df.head(10), use_container_width=True)
st.caption(f"{len(raw_df)} row(s), {len(raw_df.columns)} column(s) detected.")

report = validate_upload(raw_df)

st.markdown("### Validation Summary")

if report.errors:
    for err in report.errors:
        st.error(err)
    st.stop()

for warn in report.warnings:
    st.warning(warn)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Records detected", report.n_records)
    st.metric("Duplicate rows", report.duplicate_rows)
with col2:
    date_range_label = f"{report.date_range[0]} – {report.date_range[1]}" if report.date_range else "N/A"
    st.metric("Date range", date_range_label)
    st.metric("Missing values", report.missing_values)
with col3:
    st.metric("Historical coverage", f"{report.coverage_months:.1f} months")
    st.metric("Forecast readiness", report.forecast_readiness)

st.markdown(f"#### Data Quality Score — {quality_score_badge(report.quality_score)}")
with st.expander("How is this score calculated?"):
    st.markdown(
        """
The score is out of 100, split transparently across six factors:

- **Completeness** (30 pts) — share of non-missing values across required + provided optional columns.
- **No duplicates** (15 pts) — penalises duplicate rows / duplicate (school, date) pairs.
- **Date validity** (15 pts) — penalises rows whose date could not be parsed.
- **Historical coverage** (20 pts) — full marks at 12+ months of date range covered.
- **Column richness** (10 pts) — how many optional columns (weather, cost, attendance, etc.) are present.
- **Numeric validity** (10 pts) — penalises negative or non-numeric values in fields that must be non-negative.
"""
    )
    breakdown_df = pd.DataFrame(
        [{"Factor": k.replace("_", " ").title(), "Points": v} for k, v in report.score_breakdown.items()]
    )
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

if report.optional_columns_missing:
    st.caption(
        "Optional columns not provided (forecasting/decision-support features relying on these will be "
        f"unavailable): {', '.join(report.optional_columns_missing)}"
    )

st.info(
    f"**Forecast readiness: {report.forecast_readiness}.** "
    + (
        "Enough history to evaluate the existing model against this data."
        if report.forecast_readiness == "READY"
        else "Limited history — treat any forecast on this data as indicative only, not reliable."
        if report.forecast_readiness == "LIMITED"
        else "Not enough valid data to proceed."
    )
)

compat_issues = check_feature_compatibility(raw_df)
if compat_issues:
    st.warning(
        "The existing trained model may not apply cleanly to this dataset:\n\n"
        + "\n".join(f"- {issue}" for issue in compat_issues)
    )

st.markdown("### Process Dataset")
st.caption(
    "This runs the same cleaning + feature-engineering pipeline used for the development dataset. "
    "It does **not** overwrite the synthetic dataset and does **not** retrain the model."
)

if st.button("Process Dataset", type="primary"):
    try:
        cleaned = run_preprocessing(raw_df)
    except SchemaValidationError as exc:
        st.error(f"Schema validation failed during processing: {exc}")
        st.stop()

    saved_path = save_uploaded_dataset(raw_df, uploaded_file.name)
    st.success(f"Dataset processed and saved to `{saved_path.relative_to(saved_path.parent.parent.parent)}`.")

    try:
        engineered = engineer_features(cleaned)
        st.markdown("#### Processed & Feature-Engineered Preview")
        st.dataframe(engineered.head(20), use_container_width=True)
        st.caption(
            f"{len(engineered)} row(s) retained after feature engineering "
            f"(rows lacking full 14-day lag history or a meals_served value are dropped)."
        )
    except Exception as exc:
        st.warning(
            "Cleaning succeeded, but feature engineering could not run on this dataset "
            f"(commonly caused by missing columns needed for lag/rolling features): {exc}"
        )
        st.markdown("#### Cleaned Data Preview")
        st.dataframe(cleaned.head(20), use_container_width=True)
