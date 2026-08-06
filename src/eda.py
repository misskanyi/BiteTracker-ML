"""Statistical summaries and data-quality reporting for the processed dataset.

Separate from visualization.py (charts) so numeric/report logic can be
tested and reused independently of Plotly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import settings
from src.utils import get_logger

logger = get_logger(__name__)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return `.describe()` for all numeric columns."""
    return df.select_dtypes(include="number").describe().T


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column."""
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def duplicate_report(df: pd.DataFrame) -> dict[str, int]:
    """Count exact-row duplicates and duplicate (school_id, date) pairs."""
    return {
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_school_date_pairs": int(
            df.duplicated(subset=[settings.SCHOOL_ID_COLUMN, settings.DATE_COLUMN]).sum()
        ),
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix across numeric columns."""
    return df.select_dtypes(include="number").corr()


def categorical_summary(df: pd.DataFrame) -> dict[str, dict]:
    """Value counts for the dataset's categorical columns."""
    cat_cols = ["school_name", "county", "sub_county", "school_term", "day_of_week"]
    return {col: df[col].value_counts().to_dict() for col in cat_cols if col in df.columns}


def detect_outlier_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    """Count IQR-fence outliers per column (informational — not a mutation)."""
    counts = {}
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def build_data_quality_report(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    """Summarize dataset shape, quality issues, and preprocessing performed."""
    return {
        "raw_record_count": int(len(raw_df)),
        "processed_record_count": int(len(processed_df)),
        "num_schools": int(raw_df[settings.SCHOOL_ID_COLUMN].nunique()),
        "date_range": {
            "start": str(pd.to_datetime(raw_df[settings.DATE_COLUMN]).min().date()),
            "end": str(pd.to_datetime(raw_df[settings.DATE_COLUMN]).max().date()),
        },
        "feature_count": int(len(processed_df.columns)),
        "target_variable": settings.TARGET_COLUMN,
        "missing_values_raw": int(raw_df.isna().sum().sum()),
        "missing_values_processed": int(processed_df.isna().sum().sum()),
        "duplicates_raw": duplicate_report(raw_df),
        "outliers_detected_raw": detect_outlier_counts(
            raw_df, ["meals_served", "leftover_meals", "enrolled_students"]
        ),
        "preprocessing_steps_applied": [
            "schema validation",
            "date parsing",
            "dtype coercion",
            "chronological sort (per school)",
            "duplicate removal",
            "non-negative value enforcement",
            "per-school missing-value imputation (ffill/bfill + median fallback)",
            "per-school IQR outlier capping",
            "one-hot encoding of school_term and day_of_week",
            "lag/rolling feature engineering (leakage-safe, per school)",
            "next-day target construction",
            "incomplete-row removal (insufficient history / no target)",
        ],
    }


def save_reports(raw_df: pd.DataFrame, processed_df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Compute and persist all statistical/quality reports to reports/metrics/."""
    out_dir = output_dir or settings.METRICS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptive_statistics(processed_df).to_csv(out_dir / "descriptive_statistics.csv")
    missing_value_report(raw_df).to_csv(out_dir / "missing_value_report.csv")
    correlation_matrix(processed_df).to_csv(out_dir / "correlation_matrix.csv")

    with open(out_dir / "categorical_summary.json", "w") as f:
        json.dump(categorical_summary(processed_df), f, indent=2, default=str)

    quality_report = build_data_quality_report(raw_df, processed_df)
    with open(out_dir / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)

    logger.info("Saved statistical and data-quality reports to %s", out_dir)
    return out_dir
