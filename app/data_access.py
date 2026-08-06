"""Cached data/model loading helpers shared by every Streamlit page."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import settings


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / "school_meals_processed.csv"
    df = pd.read_csv(path)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(settings.SYNTHETIC_DATA_PATH)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df


@st.cache_resource(show_spinner=False)
def load_final_model():
    return joblib.load(settings.MODELS_DIR / "final_best_model.joblib")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def artifacts_ready() -> dict[str, bool]:
    """Check which pipeline outputs exist, so pages can degrade gracefully."""
    return {
        "processed_data": (settings.PROCESSED_DATA_DIR / "school_meals_processed.csv").exists(),
        "baseline_metrics": (settings.METRICS_DIR / "baseline_model_metrics.csv").exists(),
        "tuned_metrics": (settings.METRICS_DIR / "tuned_model_metrics.csv").exists(),
        "final_model": (settings.MODELS_DIR / "final_best_model.joblib").exists(),
        "final_model_metadata": (settings.MODELS_DIR / "final_best_model_metadata.json").exists(),
    }
