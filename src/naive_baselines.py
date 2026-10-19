"""Naive forecasting baselines: simple rules the ML models have to beat.

Both use only each school's own realised meal demand (`meals_served`) from
observations strictly before the target date T:

- Persistence / Last Observation: prediction(T) = meals served on that
  school's most recent available observation before T.
- 7-Day Historical Mean: prediction(T) = mean meals served over that
  school's previous 7 available observations (T itself excluded).

History is grouped by school, sorted by date, and shifted by one row before
any window is taken, so T's own demand and any later day never enter its
prediction, and one school's history never mixes with another's. Rows with
missing `meals_served` are skipped ("available"`     observations only).

Neither rule is tuned or fit to data, and neither uses `projected_students`.
"""

# In simple terms: two "no-ML" guesses — "same as last time" and "average
# of the last 7 days" — used to check whether the ML models are worth it.

from __future__ import annotations

import pandas as pd

from config import settings
from src.model_evaluation import compute_regression_metrics

PERSISTENCE = "Persistence / Last Observation"
HISTORICAL_MEAN_7 = "7-Day Historical Mean"
NAIVE_BASELINES = (PERSISTENCE, HISTORICAL_MEAN_7)
HISTORICAL_MEAN_WINDOW = 7

PREDICTION_COLUMNS = {
    PERSISTENCE: "persistence_prediction",
    HISTORICAL_MEAN_7: "historical_mean_7_prediction",
}


def compute_naive_forecasts(history: pd.DataFrame, demand_column: str = "meals_served") -> pd.DataFrame:
    """Return one row per available (school, date) observation with both naive forecasts.

    `history` is the full observed history (every school, every date). The
    output keeps school_id, date and the actual demand, plus one column per
    baseline (NaN where not enough prior history exists).
    """
    school, date = settings.SCHOOL_ID_COLUMN, settings.DATE_COLUMN
    obs = history[[school, date, demand_column]].dropna(subset=[demand_column])
    if obs.duplicated(subset=[school, date]).any():
        raise ValueError("History has duplicate (school_id, date) rows; naive forecasts would be ambiguous")
    obs = obs.sort_values([school, date]).reset_index(drop=True)

    previous = obs.groupby(school)[demand_column].shift(1)
    obs[PREDICTION_COLUMNS[PERSISTENCE]] = previous
    obs[PREDICTION_COLUMNS[HISTORICAL_MEAN_7]] = previous.groupby(obs[school]).transform(
        lambda s: s.rolling(window=HISTORICAL_MEAN_WINDOW, min_periods=HISTORICAL_MEAN_WINDOW).mean()
    )
    return obs


def naive_forecasts_for_period(history: pd.DataFrame, period_df: pd.DataFrame) -> pd.DataFrame:
    """Naive forecasts for exactly the (school, date) rows in `period_df` (e.g. the test set).

    Forecasts are computed on the full `history`, so the first test day's
    persistence forecast is the last training-period observation — the same
    information an operator would have had that morning.
    """
    school, date = settings.SCHOOL_ID_COLUMN, settings.DATE_COLUMN
    forecasts = compute_naive_forecasts(history)
    keys = period_df[[school, date]]
    merged = keys.merge(
        forecasts[[school, date, *PREDICTION_COLUMNS.values()]], on=[school, date], how="left", validate="1:1"
    )
    merged.index = period_df.index
    missing = merged[list(PREDICTION_COLUMNS.values())].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} evaluation rows lack enough prior history for a naive forecast"
        )
    return merged


def evaluate_naive_baselines(history: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score both naive baselines on the test period.

    Returns (metrics_df, predictions_df). Metrics use the same MAE/RMSE/R²
    definitions as the ML models, against the same target column.
    """
    forecasts = naive_forecasts_for_period(history, test_df)
    y_test = test_df[settings.TARGET_COLUMN]
    rows = []
    for name in NAIVE_BASELINES:
        metrics = compute_regression_metrics(y_test, forecasts[PREDICTION_COLUMNS[name]])
        rows.append(
            {
                "Method": name,
                "Type": "Naive baseline",
                "Test MAE": metrics["MAE"],
                "Test RMSE": metrics["RMSE"],
                "Test R2": metrics["R2"],
            }
        )
    predictions = pd.DataFrame(
        {
            "date": test_df[settings.DATE_COLUMN].values,
            "school_id": test_df[settings.SCHOOL_ID_COLUMN].values,
            "actual_meals": y_test.values,
            **{col: forecasts[col].values for col in PREDICTION_COLUMNS.values()},
        }
    )
    return pd.DataFrame(rows), predictions


def compare_to_naive_baselines(model_name: str, model_metrics: dict, naive_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Selected-model-vs-baseline differences on the test period.

    MAE improvement = baseline MAE - model MAE (positive = model is better).
    Percentage improvement is left empty when the baseline MAE is 0.
    """
    rows = []
    for _, base in naive_metrics_df.iterrows():
        improvement = base["Test MAE"] - model_metrics["MAE"]
        pct = improvement / base["Test MAE"] * 100 if base["Test MAE"] > 0 else None
        rows.append(
            {
                "Selected Model": model_name,
                "Naive Baseline": base["Method"],
                "Baseline Test MAE": base["Test MAE"],
                "Model Test MAE": model_metrics["MAE"],
                "MAE Improvement": improvement,
                "MAE Improvement (%)": pct,
                "Baseline Test RMSE": base["Test RMSE"],
                "Model Test RMSE": model_metrics["RMSE"],
                "RMSE Improvement": base["Test RMSE"] - model_metrics["RMSE"],
                "Model Beats Baseline (MAE)": bool(improvement > 0),
            }
        )
    return pd.DataFrame(rows)
