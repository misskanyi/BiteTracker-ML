"""CLI entry point: baseline model training (Phase 4).

Trains Linear Regression, Decision Tree, Random Forest, and XGBoost with
default hyperparameters on a chronological 80/20 split, saves models and
metrics, and writes reports/Baseline_Model_Training_Report.md.

Usage:
    python run_training.py
"""

from __future__ import annotations

import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.model_evaluation import compute_regression_metrics
from src.model_training import (
    TRAINERS,
    chronological_split,
    prepare_features_and_target,
    save_model,
)
from src.utils import get_logger

logger = get_logger(__name__)

PRETTY_NAMES = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def load_processed_dataset() -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / "school_meals_processed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run `python run_pipeline.py` first."
        )
    df = pd.read_csv(path)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 80/20 split, with a check that train never overlaps the test period."""
    train_df, test_df = chronological_split(df)

    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df), train_df[settings.DATE_COLUMN].min().date(), train_df[settings.DATE_COLUMN].max().date(),
        len(test_df), test_df[settings.DATE_COLUMN].min().date(), test_df[settings.DATE_COLUMN].max().date(),
    )
    if train_df[settings.DATE_COLUMN].max() >= test_df[settings.DATE_COLUMN].min():
        raise ValueError("Train period must end before the test period starts (no shared dates)")
    return train_df, test_df


def train_and_evaluate_baselines(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit every model in TRAINERS with default settings and score it on train and test.

    Returns (metrics_df, test_predictions, trained_models). Nothing is written to disk.
    """
    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    results = []
    predictions = pd.DataFrame(
        {
            "date": test_df[settings.DATE_COLUMN].values,
            "school_id": test_df[settings.SCHOOL_ID_COLUMN].values,
            "actual_meals": y_test.values,
        }
    )

    trained_models = {}
    for name, trainer in TRAINERS.items():
        logger.info("Training %s", name)
        start = time.perf_counter()
        model = trainer(X_train, y_train)
        train_seconds = time.perf_counter() - start

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = compute_regression_metrics(y_train, train_pred)
        test_metrics = compute_regression_metrics(y_test, test_pred)

        results.append(
            {
                "Model": PRETTY_NAMES[name],
                "Train MAE": train_metrics["MAE"],
                "Test MAE": test_metrics["MAE"],
                "Train RMSE": train_metrics["RMSE"],
                "Test RMSE": test_metrics["RMSE"],
                "Train R2": train_metrics["R2"],
                "Test R2": test_metrics["R2"],
                "Training Time (s)": train_seconds,
            }
        )

        predictions[f"{name}_prediction"] = test_pred
        trained_models[name] = model

    return pd.DataFrame(results), predictions, trained_models


def save_baseline_outputs(
    metrics_df: pd.DataFrame,
    predictions: pd.DataFrame,
    trained_models: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Save baseline models, metrics, predictions, split metadata, charts and report."""
    for name, model in trained_models.items():
        save_model(model, name)

    metrics_path = settings.METRICS_DIR / "baseline_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved baseline metrics to %s", metrics_path)

    predictions_path = settings.METRICS_DIR / "baseline_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    logger.info("Saved test predictions to %s", predictions_path)

    with open(settings.MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    split_metadata = {
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "train_date_range": {
            "start": str(train_df[settings.DATE_COLUMN].min().date()),
            "end": str(train_df[settings.DATE_COLUMN].max().date()),
        },
        "test_date_range": {
            "start": str(test_df[settings.DATE_COLUMN].min().date()),
            "end": str(test_df[settings.DATE_COLUMN].max().date()),
        },
        "target_variable": settings.TARGET_COLUMN,
        "feature_count": len(FEATURE_COLUMNS),
        "random_seed": settings.RANDOM_SEED,
        "train_test_split_ratio": settings.TRAIN_TEST_SPLIT_RATIO,
    }
    with open(settings.MODELS_DIR / "train_test_split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)

    generate_charts(metrics_df, predictions)
    write_report(metrics_df, split_metadata)
    return split_metadata


def main() -> None:
    df = load_processed_dataset()
    train_df, test_df = split_train_test(df)
    metrics_df, predictions, trained_models = train_and_evaluate_baselines(train_df, test_df)
    save_baseline_outputs(metrics_df, predictions, trained_models, train_df, test_df)

    logger.info("Baseline training complete.")
    print(metrics_df.to_string(index=False))


def generate_charts(metrics_df: pd.DataFrame, predictions: pd.DataFrame) -> None:
    figures_dir = settings.FIGURES_DIR

    # Baseline model comparison (Test MAE / RMSE / R2 grouped bar).
    comparison_long = metrics_df.melt(
        id_vars="Model", value_vars=["Test MAE", "Test RMSE", "Test R2"], var_name="Metric", value_name="Value"
    )
    fig = px.bar(comparison_long, x="Model", y="Value", color="Metric", barmode="group", template="plotly_white")
    fig.update_layout(title="Baseline Model Comparison (Test Set)", title_x=0.02)
    fig.write_html(figures_dir / "baseline_model_comparison.html", include_plotlyjs="cdn")

    # Train vs test performance (MAE and R2) per model.
    perf_long = metrics_df.melt(
        id_vars="Model", value_vars=["Train MAE", "Test MAE", "Train R2", "Test R2"],
        var_name="Metric", value_name="Value",
    )
    fig = px.bar(perf_long, x="Model", y="Value", color="Metric", barmode="group", template="plotly_white")
    fig.update_layout(title="Train vs Test Performance", title_x=0.02)
    fig.write_html(figures_dir / "train_vs_test_performance.html", include_plotlyjs="cdn")

    # Actual vs predicted + residual distribution, per model.
    for name in TRAINERS:
        pred_col = f"{name}_prediction"
        fig = px.scatter(
            predictions, x="actual_meals", y=pred_col, opacity=0.5, template="plotly_white",
            labels={"actual_meals": "Actual Meals Served", pred_col: "Predicted Meals Served"},
        )
        min_v = min(predictions["actual_meals"].min(), predictions[pred_col].min())
        max_v = max(predictions["actual_meals"].max(), predictions[pred_col].max())
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction", line=dict(dash="dash", color="gray")))
        fig.update_layout(title=f"Actual vs Predicted — {PRETTY_NAMES[name]}", title_x=0.02)
        fig.write_html(figures_dir / f"actual_vs_predicted_{name}.html", include_plotlyjs="cdn")

        residuals = predictions["actual_meals"] - predictions[pred_col]
        fig = px.histogram(residuals, nbins=50, template="plotly_white", labels={"value": "Residual (Actual - Predicted)"})
        fig.update_layout(title=f"Residual Distribution — {PRETTY_NAMES[name]}", title_x=0.02, showlegend=False)
        fig.write_html(figures_dir / f"residuals_{name}.html", include_plotlyjs="cdn")

    logger.info("Saved baseline training charts to %s", figures_dir)


def _overfitting_note(row: pd.Series) -> str:
    mae_gap_pct = (row["Test MAE"] - row["Train MAE"]) / max(row["Train MAE"], 1e-9) * 100
    r2_gap = row["Train R2"] - row["Test R2"]
    if mae_gap_pct > 50 or r2_gap > 0.15:
        return (
            f"Potential overfitting: test MAE is {mae_gap_pct:.0f}% higher than train MAE and "
            f"train R² exceeds test R² by {r2_gap:.3f}. The model may be fitting training-period "
            "noise that doesn't generalize to the held-out period."
        )
    if mae_gap_pct < 10 and abs(r2_gap) < 0.05:
        return "Train/test performance are close — no strong evidence of overfitting."
    return (
        f"Moderate generalization gap (test MAE {mae_gap_pct:.0f}% higher than train MAE, "
        f"R² gap {r2_gap:.3f}) — worth monitoring but not a clear overfitting signal on its own."
    )


def write_report(metrics_df: pd.DataFrame, split_metadata: dict) -> None:
    best_by_mae = metrics_df.loc[metrics_df["Test MAE"].idxmin(), "Model"]
    best_by_rmse = metrics_df.loc[metrics_df["Test RMSE"].idxmin(), "Model"]
    best_by_r2 = metrics_df.loc[metrics_df["Test R2"].idxmax(), "Model"]

    lines = [
        "# BiteTracker — Baseline Model Training Report",
        "",
        "> **SYNTHETIC DEVELOPMENT RESULTS.** Results below are from **synthetic development data** "
        "(`data/sample/synthetic_school_meals.csv`), not Food4Education records, and are measured, not fabricated — "
        "they will change once a real dataset is used.",
        "",
        "## Training Process",
        "",
        "Four ML regressors were trained with untuned, library-default hyperparameters "
        "(no GridSearchCV — that is Phase 5) as a reference point before any tuning effort. "
        "These are default-hyperparameter ML models, not naive forecasting baselines: the "
        "naive baselines (Persistence and 7-Day Historical Mean) are reported in "
        "`Hyperparameter_Tuning_Report.md`. Test-set results here are descriptive only; the "
        "final model is selected in Phase 5 by TimeSeriesSplit validation MAE.",
        "",
        "- **Linear Regression** — simple, interpretable ML candidate; assumes linear relationships "
        "between lag/rolling/weather features and a school day's demand.",
        "- **Decision Tree Regressor** — captures non-linear splits (e.g. holiday vs term-time "
        "thresholds) a linear model can't.",
        "- **Random Forest Regressor** — ensemble of trees, generally more robust and less prone "
        "to overfitting than a single tree.",
        "- **XGBoost Regressor** — gradient-boosted trees, often strong on tabular data with mixed "
        "categorical/numeric/lag features like this dataset.",
        "",
        "## Chronological Split",
        "",
        f"- Train records: **{split_metadata['train_records']}** "
        f"({split_metadata['train_date_range']['start']} to {split_metadata['train_date_range']['end']})",
        f"- Test records: **{split_metadata['test_records']}** "
        f"({split_metadata['test_date_range']['start']} to {split_metadata['test_date_range']['end']})",
        f"- Split ratio: **{split_metadata['train_test_split_ratio']:.0%}** train / "
        f"**{1 - split_metadata['train_test_split_ratio']:.0%}** test, by row order after date-sorting "
        "(no shuffling).",
        "- The test set is strictly the last 20% of chronologically-ordered observations, so every "
        "test-set prediction is made using only information from earlier in time.",
        "",
        "## Baseline Results (Test Set)",
        "",
        metrics_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"- Best by Test MAE: **{best_by_mae}**",
        f"- Best by Test RMSE: **{best_by_rmse}**",
        f"- Best by Test R²: **{best_by_r2}**",
        "",
        "## Generalization / Overfitting Check",
        "",
    ]

    for _, row in metrics_df.iterrows():
        lines.append(f"- **{row['Model']}**: {_overfitting_note(row)}")

    lines += [
        "",
        "## Why Tuning Is Still Required",
        "",
        "These are default-hyperparameter baselines. Tree-based models in particular "
        "(Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and "
        "learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` "
        "to search a small, practical hyperparameter grid per model without shuffling or "
        "leaking future data into cross-validation folds. Whichever model looks best here "
        "may or may not remain best after tuning.",
        "",
        "## Synthetic Data Caveat",
        "",
        "All figures in this report come from the synthetic dataset generated for development "
        "(see `src/synthetic_data.py`), not real school records. Relationships (e.g. rainfall's "
        "effect on attendance) were designed to be plausible but are not calibrated to any real "
        "county's data. Treat these results as a pipeline validation exercise, not a claim about "
        "real-world model performance.",
        "",
    ]

    out_path = settings.REPORTS_DIR / "Baseline_Model_Training_Report.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
