"""Model-performance figures, read from the artifacts the training pipeline saved.

Every number comes from `models/final_best_model_metadata.json` or
`reports/metrics/*.csv`; nothing is recomputed or estimated here. The
results label stored by the pipeline ("SYNTHETIC DEVELOPMENT RESULTS" for
the current dataset) is passed through unchanged.
"""

from __future__ import annotations

import math

import pandas as pd

from config import settings
from services import ServiceError
from services.model_service import load_model_metadata

NAIVE_METRICS_FILENAME = "naive_baseline_metrics.csv"
VALIDATION_RANKING_FILENAME = "validation_model_selection.csv"

PERSISTENCE_BASELINE = "Persistence / Last Observation"
SEVEN_DAY_MEAN_BASELINE = "7-Day Historical Mean"
REQUIRED_BASELINES = (PERSISTENCE_BASELINE, SEVEN_DAY_MEAN_BASELINE)


class PerformanceDataError(ServiceError):
    """A required performance artifact is missing or incomplete."""


def _read_csv(filename: str) -> pd.DataFrame:
    path = settings.METRICS_DIR / filename
    if not path.exists():
        raise PerformanceDataError(
            f"Metrics file not found at {path}. Run `python scripts/train_models.py` to generate it."
        )
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise PerformanceDataError(f"Metrics file at {path} could not be read: {exc}") from exc


def _clean(value):
    """CSV/JSON scalars → plain Python (NaN → None)."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def get_naive_baselines() -> dict[str, dict]:
    """Test-period metrics of the Persistence and 7-Day Historical Mean baselines."""
    df = _read_csv(NAIVE_METRICS_FILENAME)
    required_columns = {"Method", "Test MAE", "Test RMSE", "Test R2"}
    if not required_columns <= set(df.columns):
        raise PerformanceDataError(f"{NAIVE_METRICS_FILENAME} is missing columns: {sorted(required_columns - set(df.columns))}.")
    baselines = {
        row["Method"]: {
            "name": row["Method"],
            "test_mae": _clean(row["Test MAE"]),
            "test_rmse": _clean(row["Test RMSE"]),
            "test_r2": _clean(row["Test R2"]),
        }
        for _, row in df.iterrows()
    }
    missing = [name for name in REQUIRED_BASELINES if name not in baselines]
    if missing:
        raise PerformanceDataError(f"{NAIVE_METRICS_FILENAME} has no results for: {', '.join(missing)}.")
    return baselines


def get_validation_ranking() -> list[dict]:
    """Every candidate's TimeSeriesSplit validation scores, in selection-rank order (may be empty)."""
    path = settings.METRICS_DIR / VALIDATION_RANKING_FILENAME
    if not path.exists():
        return []
    df = _read_csv(VALIDATION_RANKING_FILENAME)
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def get_model_performance() -> dict:
    """Everything a Model Performance page needs, as plain Python values."""
    metadata = load_model_metadata()
    baselines = get_naive_baselines()

    # Model-vs-baseline comparison exactly as the pipeline saved it.
    saved_comparison = {row.get("Naive Baseline"): row for row in metadata.get("naive_baseline_comparison") or []}
    missing = [name for name in REQUIRED_BASELINES if name not in saved_comparison]
    if missing:
        raise PerformanceDataError(f"Model metadata has no baseline comparison for: {', '.join(missing)}.")
    try:
        comparison = [
            {
                "baseline": name,
                "baseline_test_mae": saved_comparison[name]["Baseline Test MAE"],
                "model_test_mae": saved_comparison[name]["Model Test MAE"],
                "mae_improvement": saved_comparison[name]["MAE Improvement"],
                "mae_improvement_pct": saved_comparison[name]["MAE Improvement (%)"],
                "baseline_test_rmse": saved_comparison[name]["Baseline Test RMSE"],
                "model_test_rmse": saved_comparison[name]["Model Test RMSE"],
                "rmse_improvement": saved_comparison[name]["RMSE Improvement"],
                "model_beats_baseline": saved_comparison[name]["Model Beats Baseline (MAE)"],
            }
            for name in REQUIRED_BASELINES
        ]
    except KeyError as exc:
        raise PerformanceDataError(f"Model metadata baseline comparison is missing field {exc}.") from exc
    test_mae = metadata["final_test_mae"]

    return {
        "results_label": metadata["results_label"],
        "synthetic_data_warning": metadata.get("synthetic_data_warning"),
        "selected_model": metadata["selected_model"],
        "model_kind": metadata.get("model_kind"),
        "selection": {
            "criterion": metadata.get("selection_criterion"),
            "tiebreak": metadata.get("tiebreak_criterion"),
            "cv_strategy": metadata.get("cv_strategy"),
            "cv_n_folds": metadata.get("cv_n_folds"),
            "selection_data": metadata.get("selection_data"),
        },
        "validation": {
            "mae": metadata["validation_mae"],
            "mae_std": metadata.get("validation_mae_std"),
            "rmse": metadata.get("validation_rmse"),
        },
        "test": {
            "mae": test_mae,
            "rmse": metadata["final_test_rmse"],
            "r2": metadata["final_test_r2"],
            "evaluation": metadata.get("final_test_evaluation"),
        },
        "train_date_range": metadata.get("train_date_range"),
        "test_date_range": metadata.get("test_date_range"),
        "train_records": metadata.get("train_records"),
        "test_records": metadata.get("test_records"),
        "naive_baselines": [baselines[name] for name in REQUIRED_BASELINES],
        "baseline_comparison": comparison,
        "validation_ranking": get_validation_ranking(),
        "features_used": list(metadata["features_used"]),
        "trained_at": metadata.get("timestamp"),
    }
