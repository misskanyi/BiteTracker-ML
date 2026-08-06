"""BiteTracker — Data Explorer page: browse, filter, and download the dataset."""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
if str(_APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_APP_DIR.parent))

import streamlit as st
from theme import configure_page, page_header

from config import settings
from data_access import artifacts_ready, load_raw_data

configure_page("Data Explorer", icon="🔍")
page_header("Data Explorer", subtitle="Browse, filter, and download the underlying dataset.", icon="🔍")

if not artifacts_ready()["processed_data"]:
    st.error("Dataset not found. Run `python run_pipeline.py` first.")
    st.stop()

df = load_raw_data()

st.markdown("### Filters")
col1, col2 = st.columns(2)
with col1:
    schools = sorted(df["school_name"].unique())
    selected_schools = st.multiselect("School", schools, default=schools)
with col2:
    min_date, max_date = df[settings.DATE_COLUMN].min().date(), df[settings.DATE_COLUMN].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df[df["school_name"].isin(selected_schools)] if selected_schools else df
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered[settings.DATE_COLUMN].dt.date >= start) & (filtered[settings.DATE_COLUMN].dt.date <= end)]

st.markdown(f"### Dataset Preview ({len(filtered):,} rows)")
st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Summary Statistics")
    st.dataframe(filtered.select_dtypes(include="number").describe().T, use_container_width=True)

with col4:
    st.markdown("### Missing Value Summary")
    missing = filtered.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the current filtered selection.")
    else:
        missing_df = missing.rename("missing_count").reset_index().rename(columns={"index": "column"})
        missing_df["missing_pct"] = (missing_df["missing_count"] / len(filtered) * 100).round(2)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

st.divider()
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV", data=csv_bytes, file_name="bitetracker_filtered.csv", mime="text/csv",
)
