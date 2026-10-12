"""CLI entry point: Phase 5 hyperparameter tuning and final model selection.

Order of operations (the final test period never influences which model wins):

1. Tune Decision Tree, Random Forest and XGBoost with TimeSeriesSplit +
   GridSearchCV/RandomizedSearchCV on the training period only.
2. Cross-validate Linear Regression (not tuned) on the same folds.
3. Select the final model by lowest TimeSeriesSplit validation MAE
   (validation RMSE as tiebreak) — `select_final_model` takes validation
   scores only.
4. Evaluate the selected model once on the held-out test period.
5. For the research comparison tables only, also score the other models and
   the two naive forecasting baselines (src/naive_baselines.py) on the test
   period. These numbers are produced after selection and are not used by it.

Usage:
    python run_tuning.py
"""

from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import xgboost

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.model_evaluation import classify_generalization_gap, compute_regression_metrics
from src.model_selection import (
    SELECTION_CRITERION,
    TIEBREAK_CRITERION,
    ValidationScore,
    cross_validate_linear_regression,
    select_model_by_validation,
    validation_score_from_search,
)
from src.model_training import chronological_split, prepare_features_and_target, train_linear_regression
from src.model_tuning import (
    TUNERS,
    describe_cv_folds,
    get_time_series_cv,
    save_best_hyperparameters,
    save_tuned_model,
)
from src.naive_baselines import compare_to_naive_baselines, evaluate_naive_baselines
from src.utils import get_logger

logger = get_logger(__name__)

PRETTY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
SHORT_NAMES = {pretty: short for short, pretty in PRETTY_NAMES.items()}

SYNTHETIC_LABEL = "SYNTHETIC DEVELOPMENT RESULTS"
REAL_DATA_LABEL = "FOOD4EDUCATION RESEARCH RESULTS"


def results_label() -> str:
    """Label for saved results: real only when the Food4Education Google Sheet is the data source."""
    return REAL_DATA_LABEL if settings.USE_GOOGLE_SHEET else SYNTHETIC_LABEL

# Phase 4 default-hyperparameter results — used only for the untuned-vs-tuned
# comparison, recomputed from the saved metrics file (not hardcoded/fabricated).
BASELINE_METRICS_PATH = settings.METRICS_DIR / "baseline_model_metrics.csv"


def load_processed_dataset() -> pd.DataFrame:
    path = settings.PROCESSED_DATA_DIR / "school_meals_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run `python run_pipeline.py` first.")
    df = pd.read_csv(path)
    df[settings.DATE_COLUMN] = pd.to_datetime(df[settings.DATE_COLUMN])
    return df


def load_baseline_metrics() -> pd.DataFrame:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found at {BASELINE_METRICS_PATH}. Run `python run_training.py` first."
        )
    return pd.read_csv(BASELINE_METRICS_PATH)


def tune_models(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> tuple[dict, list[dict]]:
    """Run every tuner in TUNERS with TimeSeriesSplit on the training data only.

    Returns ({name: (fitted_search, tuning_seconds)}, cv_fold_info).
    """
    cv = get_time_series_cv(n_splits)
    fold_info = describe_cv_folds(len(X_train), n_splits)
    logger.info("TimeSeriesSplit folds (training data only, %d rows):", len(X_train))
    for fold in fold_info:
        logger.info(
            "  Fold %d: train rows %d (idx %s), val rows %d (idx %s)",
            fold["fold"], fold["train_rows"], fold["train_range"], fold["val_rows"], fold["val_range"],
        )

    searches = {}
    for name, tuner in TUNERS.items():
        pretty = PRETTY_NAMES[name]
        logger.info("Tuning %s ...", pretty)
        start = time.perf_counter()
        search = tuner(X_train, y_train, cv)
        tuning_seconds = time.perf_counter() - start
        logger.info("%s tuning finished in %.1fs — best CV MAE: %.3f", pretty, tuning_seconds, -search.best_score_)
        searches[name] = (search, tuning_seconds)
    return searches, fold_info


def collect_validation_scores(
    searches: dict, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5
) -> list[ValidationScore]:
    """TimeSeriesSplit validation scores for all four candidates, training period only.

    Tuned models report the CV score of their best hyperparameters; Linear
    Regression (no grid) is cross-validated once on the same folds.
    """
    cv = get_time_series_cv(n_splits)
    scores = [cross_validate_linear_regression(X_train, y_train, cv)]
    for name, (search, _) in searches.items():
        scores.append(validation_score_from_search(PRETTY_NAMES[name], search))
    for score in scores:
        rmse = f"{score.cv_rmse:.3f}" if score.cv_rmse is not None else "n/a"
        logger.info(
            "%s: validation MAE %.3f (± %.3f over %d folds), validation RMSE %s",
            score.model, score.cv_mae, score.cv_mae_std, score.n_folds, rmse,
        )
    return scores


def select_final_model(validation_scores: list[ValidationScore]) -> tuple[ValidationScore, list[ValidationScore]]:
    """Final model selection by lowest TimeSeriesSplit validation MAE (validation RMSE tiebreak).

    Accepts training-period validation scores only — no test data, no test
    metrics. Returns (selected, ranked).
    """
    return select_model_by_validation(validation_scores)


def fit_selected_model(selected: ValidationScore, searches: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Return (fitted_model, model_kind) for the selected candidate, trained on the full training period."""
    if selected.model == "Linear Regression":
        return train_linear_regression(X_train, y_train), "untuned"
    search, _ = searches[SHORT_NAMES[selected.model]]
    return search.best_estimator_, "tuned"


def evaluate_selected_model(
    selected: ValidationScore,
    model,
    model_kind: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the already-selected model once on the held-out test period."""
    train_metrics = compute_regression_metrics(y_train, model.predict(X_train))
    test_metrics = compute_regression_metrics(y_test, model.predict(X_test))
    logger.info(
        "Final test evaluation of %s: MAE %.3f, RMSE %.3f, R² %.3f",
        selected.model, test_metrics["MAE"], test_metrics["RMSE"], test_metrics["R2"],
    )
    return {
        "Model": selected.model,
        "model_kind": model_kind,
        "Validation MAE": selected.cv_mae,
        "Validation MAE Std": selected.cv_mae_std,
        "Validation RMSE": selected.cv_rmse,
        "CV Folds": selected.n_folds,
        "Train MAE": train_metrics["MAE"],
        "Train RMSE": train_metrics["RMSE"],
        "Train R2": train_metrics["R2"],
        "Test MAE": test_metrics["MAE"],
        "Test RMSE": test_metrics["RMSE"],
        "Test R2": test_metrics["R2"],
        "Generalization Gap Status": classify_generalization_gap(
            train_metrics["MAE"], test_metrics["MAE"], train_metrics["R2"], test_metrics["R2"]
        ),
    }


def evaluate_tuned_models(
    searches: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> dict:
    """Research comparison only: score every tuned model on train and the test period.

    Called after the final model has been selected; its output is not an
    input to `select_final_model`. `baseline_df` is the default-hyperparameter
    metrics table indexed by pretty model name. Nothing is written to disk.
    """
    tuned_metrics_rows = []
    comparison_rows = []
    best_params_all = {}
    model_metadata = {}
    best_models = {}
    predictions = pd.DataFrame(
        {
            "date": test_df[settings.DATE_COLUMN].values,
            "school_id": test_df[settings.SCHOOL_ID_COLUMN].values,
            "actual_meals": y_test.values,
        }
    )

    for name, (search, tuning_seconds) in searches.items():
        pretty = PRETTY_NAMES[name]
        best_model = search.best_estimator_
        best_models[name] = best_model
        best_params_all[name] = search.best_params_
        validation = validation_score_from_search(pretty, search)

        pred_start = time.perf_counter()
        test_pred = best_model.predict(X_test)
        prediction_seconds = time.perf_counter() - pred_start
        train_pred = best_model.predict(X_train)

        train_metrics = compute_regression_metrics(y_train, train_pred)
        test_metrics = compute_regression_metrics(y_test, test_pred)
        gap_status = classify_generalization_gap(
            train_metrics["MAE"], test_metrics["MAE"], train_metrics["R2"], test_metrics["R2"]
        )

        row = {
            "Model": pretty,
            "Train MAE": train_metrics["MAE"],
            "Test MAE": test_metrics["MAE"],
            "Train RMSE": train_metrics["RMSE"],
            "Test RMSE": test_metrics["RMSE"],
            "Train R2": train_metrics["R2"],
            "Test R2": test_metrics["R2"],
            "Generalization Gap Status": gap_status,
            "Tuning Time (s)": tuning_seconds,
            "Prediction Time (s)": prediction_seconds,
            "Best CV MAE": validation.cv_mae,
            "Best CV RMSE": validation.cv_rmse,
        }
        tuned_metrics_rows.append(row)

        baseline_row = baseline_df.loc[pretty].to_dict()
        comparison_rows.append(
            _build_comparison_row(pretty, baseline_row, row)
        )

        predictions[f"{name}_tuned_prediction"] = test_pred

        model_metadata[name] = {
            "search_type": type(search).__name__,
            "n_candidates_evaluated": len(search.cv_results_["params"]),
            "tuning_seconds": tuning_seconds,
            "best_cv_mae": validation.cv_mae,
            "best_cv_rmse": validation.cv_rmse,
        }

    return {
        "tuned_metrics_rows": tuned_metrics_rows,
        "comparison_rows": comparison_rows,
        "predictions": predictions,
        "best_params": best_params_all,
        "model_metadata": model_metadata,
        "best_models": best_models,
    }


def build_model_comparison(
    ranked: list[ValidationScore],
    final_best: dict,
    tuned_metrics_rows: list[dict],
    baseline_df: pd.DataFrame,
    naive_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """One table: naive baselines + the four ML candidates, validation and test columns side by side.

    Selection Rank comes from the validation ranking only. Test columns for
    non-selected models are for research comparison and played no part in selection.
    """
    test_by_model = {r["Model"]: r for r in tuned_metrics_rows}
    # Linear Regression is not tuned: its test metrics are the default-hyperparameter model's.
    test_by_model["Linear Regression"] = baseline_df.loc["Linear Regression"].to_dict()

    rows = [
        {
            "Method": r["Method"],
            "Type": r["Type"],
            "Validation MAE": None,
            "Validation RMSE": None,
            "Selection Rank": None,
            "Selected": False,
            "Test MAE": r["Test MAE"],
            "Test RMSE": r["Test RMSE"],
            "Test R2": r["Test R2"],
        }
        for _, r in naive_metrics_df.iterrows()
    ]
    for rank, score in enumerate(ranked, start=1):
        test = test_by_model[score.model]
        rows.append(
            {
                "Method": score.model,
                "Type": "ML candidate",
                "Validation MAE": score.cv_mae,
                "Validation RMSE": score.cv_rmse,
                "Selection Rank": rank,
                "Selected": score.model == final_best["Model"],
                "Test MAE": test["Test MAE"],
                "Test RMSE": test["Test RMSE"],
                "Test R2": test["Test R2"],
            }
        )
    return pd.DataFrame(rows)


def save_tuning_outputs(
    evaluation: dict,
    fold_info: list[dict],
    final_best: dict,
    final_model,
    ranked: list[ValidationScore],
    naive_metrics_df: pd.DataFrame,
    naive_predictions: pd.DataFrame,
    baseline_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
    extra_metadata: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save tuned models, validation scores, naive-baseline results, metrics, the final model + metadata, charts and report.

    `extra_metadata` (data source, target definition, fitted preprocessing)
    is merged into final_best_model_metadata.json.
    """
    feature_columns = list(X_test.columns)
    for name, model in evaluation["best_models"].items():
        save_tuned_model(model, name)

    tuned_metrics_df = pd.DataFrame(evaluation["tuned_metrics_rows"])
    tuned_metrics_path = settings.METRICS_DIR / "tuned_model_metrics.csv"
    tuned_metrics_df.to_csv(tuned_metrics_path, index=False)
    logger.info("Saved tuned metrics to %s", tuned_metrics_path)

    comparison_df = pd.DataFrame(evaluation["comparison_rows"])
    comparison_path = settings.METRICS_DIR / "baseline_vs_tuned_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved baseline-vs-tuned comparison to %s", comparison_path)

    predictions_path = settings.METRICS_DIR / "tuned_test_predictions.csv"
    evaluation["predictions"].to_csv(predictions_path, index=False)

    validation_df = pd.DataFrame(
        [
            {
                "Model": s.model,
                "Validation MAE": s.cv_mae,
                "Validation MAE Std": s.cv_mae_std,
                "Validation RMSE": s.cv_rmse,
                "CV Folds": s.n_folds,
                "Selection Rank": rank,
                "Selected": s.model == final_best["Model"],
            }
            for rank, s in enumerate(ranked, start=1)
        ]
    )
    validation_df.to_csv(settings.METRICS_DIR / "validation_model_selection.csv", index=False)

    naive_metrics_df.to_csv(settings.METRICS_DIR / "naive_baseline_metrics.csv", index=False)
    naive_predictions.to_csv(settings.METRICS_DIR / "naive_baseline_test_predictions.csv", index=False)
    model_comparison_df = build_model_comparison(
        ranked, final_best, evaluation["tuned_metrics_rows"], baseline_df, naive_metrics_df
    )
    model_comparison_df.to_csv(settings.METRICS_DIR / "model_comparison.csv", index=False)
    naive_comparison_df = compare_to_naive_baselines(
        final_best["Model"],
        {"MAE": final_best["Test MAE"], "RMSE": final_best["Test RMSE"]},
        naive_metrics_df,
    )
    naive_comparison_df.to_csv(settings.METRICS_DIR / "selected_model_vs_naive_baselines.csv", index=False)
    logger.info("Saved validation ranking, naive-baseline metrics and model comparison to %s", settings.METRICS_DIR)

    save_best_hyperparameters(evaluation["best_params"])
    tuning_metadata = {"n_splits": n_splits, "cv_folds": fold_info, "models": evaluation["model_metadata"]}
    with open(settings.MODELS_DIR / "tuning_metadata.json", "w") as f:
        json.dump(tuning_metadata, f, indent=2, default=str)

    save_final_best_model(
        final_best, final_model, naive_comparison_df, train_df, test_df, n_splits, feature_columns, extra_metadata
    )
    generate_charts(tuned_metrics_df, comparison_df, evaluation["predictions"], fold_info, feature_columns)
    generate_final_best_model_charts(final_best, X_test, y_test)
    write_report(
        tuned_metrics_df, comparison_df, evaluation["best_params"], fold_info, final_best, ranked,
        model_comparison_df, naive_comparison_df, n_train_rows=len(train_df),
    )
    return tuned_metrics_df, comparison_df


def main() -> None:
    df = load_processed_dataset()
    train_df, test_df = chronological_split(df)  # identical split to Phase 4
    baseline_df = load_baseline_metrics().set_index("Model")

    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    # Naive baselines use realised demand history only (no features, no fitting).
    naive_metrics_df, naive_predictions = evaluate_naive_baselines(df, test_df)

    n_splits = settings.CV_N_SPLITS
    searches, fold_info = tune_models(X_train, y_train, n_splits)
    validation_scores = collect_validation_scores(searches, X_train, y_train, n_splits)

    # Selection sees validation scores only; the test period is used after this point.
    selected, ranked = select_final_model(validation_scores)
    final_model, model_kind = fit_selected_model(selected, searches, X_train, y_train)
    final_best = evaluate_selected_model(selected, final_model, model_kind, X_train, y_train, X_test, y_test)

    evaluation = evaluate_tuned_models(searches, X_train, y_train, X_test, y_test, test_df, baseline_df)
    tuned_metrics_df, comparison_df = save_tuning_outputs(
        evaluation, fold_info, final_best, final_model, ranked, naive_metrics_df, naive_predictions,
        baseline_df, train_df, test_df, X_test, y_test, n_splits,
    )

    print(f"\n=== {results_label()} ===")
    print("\n=== Validation ranking (TimeSeriesSplit, training period) ===")
    for rank, s in enumerate(ranked, start=1):
        print(f"{rank}. {s.model}: validation MAE {s.cv_mae:.3f}")
    print("\n=== Tuned Metrics ===")
    print(tuned_metrics_df.to_string(index=False))
    print("\n=== Naive baselines (test period) ===")
    print(naive_metrics_df.to_string(index=False))
    print(
        f"\nFinal selected model: {final_best['Model']} (validation MAE={final_best['Validation MAE']:.2f}; "
        f"final test MAE={final_best['Test MAE']:.2f})"
    )


def _build_comparison_row(pretty_name: str, baseline_row: dict, tuned_row: dict) -> dict:
    baseline_status = classify_generalization_gap(
        baseline_row["Train MAE"], baseline_row["Test MAE"], baseline_row["Train R2"], baseline_row["Test R2"]
    )
    mae_improvement = baseline_row["Test MAE"] - tuned_row["Test MAE"]
    rmse_improvement = baseline_row["Test RMSE"] - tuned_row["Test RMSE"]
    r2_improvement = tuned_row["Test R2"] - baseline_row["Test R2"]
    return {
        "Model": pretty_name,
        "Baseline Test MAE": baseline_row["Test MAE"],
        "Tuned Test MAE": tuned_row["Test MAE"],
        "MAE Improvement": mae_improvement,
        "Baseline Test RMSE": baseline_row["Test RMSE"],
        "Tuned Test RMSE": tuned_row["Test RMSE"],
        "RMSE Improvement": rmse_improvement,
        "Baseline Test R2": baseline_row["Test R2"],
        "Tuned Test R2": tuned_row["Test R2"],
        "R2 Improvement": r2_improvement,
        "Baseline Overfitting Status": baseline_status,
        "Tuned Overfitting Status": tuned_row["Generalization Gap Status"],
    }


def save_final_best_model(
    final_best: dict,
    model,
    naive_comparison_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int,
    features_used: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> None:
    import joblib

    final_path = settings.MODELS_DIR / "final_best_model.joblib"
    joblib.dump(model, final_path)

    metadata = {
        "results_label": results_label(),
        "selected_model": final_best["Model"],
        "model_kind": final_best["model_kind"],
        "selection_criterion": SELECTION_CRITERION,
        "tiebreak_criterion": TIEBREAK_CRITERION,
        "selection_data": "training period only (TimeSeriesSplit validation folds); "
        "the final test period was not used for tuning or selection",
        "cv_strategy": "TimeSeriesSplit (expanding window, no shuffling) over the chronological training period",
        "cv_n_folds": n_splits,
        "validation_mae": final_best["Validation MAE"],
        "validation_mae_std": final_best["Validation MAE Std"],
        "validation_rmse": final_best["Validation RMSE"],
        "best_parameters": model.get_params() if hasattr(model, "get_params") else None,
        "train_date_range": {
            "start": str(train_df[settings.DATE_COLUMN].min().date()),
            "end": str(train_df[settings.DATE_COLUMN].max().date()),
        },
        "test_date_range": {
            "start": str(test_df[settings.DATE_COLUMN].min().date()),
            "end": str(test_df[settings.DATE_COLUMN].max().date()),
        },
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "features_used": list(FEATURE_COLUMNS if features_used is None else features_used),
        "target_variable": settings.TARGET_COLUMN,
        "final_test_evaluation": "selected model evaluated once on the held-out test period, after selection",
        "final_test_mae": final_best["Test MAE"],
        "final_test_rmse": final_best["Test RMSE"],
        "final_test_r2": final_best["Test R2"],
        "naive_baseline_comparison": naive_comparison_df.to_dict(orient="records"),
        "synthetic_data_warning": None if settings.USE_GOOGLE_SHEET else (
            "These metrics were measured on a SYNTHETIC development dataset "
            "(src/synthetic_data.py), not real school records or Food4Education data. "
            "Re-run the full pipeline and re-tune once real data is available before "
            "trusting these numbers operationally."
        ),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        **(extra_metadata or {}),
    }
    with open(settings.MODELS_DIR / "final_best_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved final best model (%s) and metadata", final_best["Model"])


def generate_charts(tuned_metrics_df, comparison_df, predictions, fold_info, feature_columns=None) -> None:
    figures_dir = settings.FIGURES_DIR
    feature_columns = list(FEATURE_COLUMNS if feature_columns is None else feature_columns)

    # Baseline vs tuned MAE / RMSE / R2
    for metric, label in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")]:
        long_df = comparison_df.melt(
            id_vars="Model", value_vars=[f"Baseline Test {metric}", f"Tuned Test {metric}"],
            var_name="Stage", value_name="Value",
        )
        fig = px.bar(long_df, x="Model", y="Value", color="Stage", barmode="group", template="plotly_white")
        fig.update_layout(title=f"Baseline vs Tuned Test {label}", title_x=0.02)
        fig.write_html(figures_dir / f"baseline_vs_tuned_{metric.lower()}.html", include_plotlyjs="cdn")

    # Train vs test after tuning
    perf_long = tuned_metrics_df.melt(
        id_vars="Model", value_vars=["Train MAE", "Test MAE", "Train R2", "Test R2"],
        var_name="Metric", value_name="Value",
    )
    fig = px.bar(perf_long, x="Model", y="Value", color="Metric", barmode="group", template="plotly_white")
    fig.update_layout(title="Train vs Test Performance (Tuned Models)", title_x=0.02)
    fig.write_html(figures_dir / "tuned_train_vs_test_performance.html", include_plotlyjs="cdn")

    # CV score comparison (best CV MAE per model)
    fig = px.bar(
        tuned_metrics_df, x="Model", y="Best CV MAE", template="plotly_white",
        title="Cross-Validation MAE by Model (TimeSeriesSplit, training data only)",
    )
    fig.update_layout(title_x=0.02)
    fig.write_html(figures_dir / "cv_score_comparison.html", include_plotlyjs="cdn")

    # Actual vs predicted + residuals per tuned model
    for name in ["decision_tree", "random_forest", "xgboost"]:
        pred_col = f"{name}_tuned_prediction"
        pretty = PRETTY_NAMES[name]
        fig = px.scatter(
            predictions, x="actual_meals", y=pred_col, opacity=0.5, template="plotly_white",
            labels={"actual_meals": "Actual Meals Served", pred_col: "Predicted Meals Served"},
        )
        min_v = min(predictions["actual_meals"].min(), predictions[pred_col].min())
        max_v = max(predictions["actual_meals"].max(), predictions[pred_col].max())
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction", line=dict(dash="dash", color="gray")))
        fig.update_layout(title=f"Actual vs Predicted (Tuned) — {pretty}", title_x=0.02)
        fig.write_html(figures_dir / f"actual_vs_predicted_tuned_{name}.html", include_plotlyjs="cdn")

        residuals = predictions["actual_meals"] - predictions[pred_col]
        fig = px.histogram(residuals, nbins=50, template="plotly_white", labels={"value": "Residual (Actual - Predicted)"})
        fig.update_layout(title=f"Residual Distribution (Tuned) — {pretty}", title_x=0.02, showlegend=False)
        fig.write_html(figures_dir / f"residuals_tuned_{name}.html", include_plotlyjs="cdn")

    # Feature importance for tree-based tuned models
    from src.model_tuning import load_tuned_model

    for name in ["decision_tree", "random_forest", "xgboost"]:
        model = load_tuned_model(name)
        if not hasattr(model, "feature_importances_"):
            continue
        importance_df = pd.DataFrame(
            {"feature": feature_columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=True)
        fig = px.bar(
            importance_df, x="importance", y="feature", orientation="h", template="plotly_white",
        )
        fig.update_layout(title=f"Feature Importance (Tuned) — {PRETTY_NAMES[name]}", title_x=0.02, height=600)
        fig.write_html(figures_dir / f"feature_importance_tuned_{name}.html", include_plotlyjs="cdn")

    # Final best model chart is generated after we know which model won — see write_report caller.
    logger.info("Saved tuning charts to %s", figures_dir)


def generate_final_best_model_charts(final_best: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    import joblib

    figures_dir = settings.FIGURES_DIR
    model = joblib.load(settings.MODELS_DIR / "final_best_model.joblib")
    preds = model.predict(X_test)

    df = pd.DataFrame({"actual_meals": y_test.values, "predicted_meals": preds})
    fig = px.scatter(
        df, x="actual_meals", y="predicted_meals", opacity=0.5, template="plotly_white",
        labels={"actual_meals": "Actual Meals Served", "predicted_meals": "Predicted Meals Served"},
    )
    min_v = min(df["actual_meals"].min(), df["predicted_meals"].min())
    max_v = max(df["actual_meals"].max(), df["predicted_meals"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Perfect Prediction", line=dict(dash="dash", color="gray")))
    fig.update_layout(title=f"Final Best Model ({final_best['Model']}) — Actual vs Predicted", title_x=0.02)
    fig.write_html(figures_dir / "final_best_model_actual_vs_predicted.html", include_plotlyjs="cdn")

    residuals = df["actual_meals"] - df["predicted_meals"]
    fig = px.histogram(residuals, nbins=50, template="plotly_white", labels={"value": "Residual (Actual - Predicted)"})
    fig.update_layout(title=f"Final Best Model ({final_best['Model']}) — Residual Distribution", title_x=0.02, showlegend=False)
    fig.write_html(figures_dir / "final_best_model_residuals.html", include_plotlyjs="cdn")
    logger.info("Saved final best model charts to %s", figures_dir)


def write_report(
    tuned_metrics_df,
    comparison_df,
    best_params_all,
    fold_info,
    final_best,
    ranked,
    model_comparison_df,
    naive_comparison_df,
    n_train_rows,
) -> None:
    baseline_summary = "; ".join(
        f"{row['Model']}: test R²={row['Baseline Test R2']:.3f} ({row['Baseline Overfitting Status']})"
        for _, row in comparison_df.iterrows()
    )
    lines = [
        "# BiteTracker — Hyperparameter Tuning Report (Phase 5)",
        "",
        (
            f"> **{REAL_DATA_LABEL}.** Results below are from the anonymised Food4Education "
            "dataset, read live from its Google Sheet. Weather, cost and exam/event fields are "
            "not in that sheet and were held constant."
            if settings.USE_GOOGLE_SHEET
            else f"> **{SYNTHETIC_LABEL}.** Results below are from **synthetic development data** "
            "(`data/sample/synthetic_school_meals.csv`), not Food4Education or any real school "
            "records. They are measured, not fabricated."
        ),
        "",
        "## Why Tuning Was Required",
        "",
        "Phase 4 models used library-default hyperparameters, which lets tree-based "
        f"models fit training noise. Measured default-hyperparameter results: {baseline_summary}. "
        "This phase tunes depth, leaf-size, regularization, and sampling parameters "
        "to reduce the train/test gap, not just to chase a lower training error.",
        "",
        "## Why TimeSeriesSplit",
        "",
        "Meal demand is a time series — a school's future demand must never be "
        "predicted using information from its own future. `TimeSeriesSplit` builds "
        "growing folds where every validation fold occurs strictly after its "
        "training fold, unlike standard k-fold which would shuffle rows and let "
        "the model \"see\" the future during cross-validation.",
        "",
        f"5 folds were used over the {n_train_rows} "
        "training rows carried over from Phase 4 (the final 20% test set was held out "
        "entirely and never passed to GridSearchCV/RandomizedSearchCV). Exact fold ranges:",
        "",
        "| Fold | Train Rows | Train Index Range | Val Rows | Val Index Range |",
        "|---|---|---|---|---|",
    ]
    for fold in fold_info:
        lines.append(
            f"| {fold['fold']} | {fold['train_rows']} | {fold['train_range']} | "
            f"{fold['val_rows']} | {fold['val_range']} |"
        )

    lines += [
        "",
        "## Scoring Strategy",
        "",
        "- **Hyperparameters are chosen by cross-validation MAE** (GridSearchCV / "
        "RandomizedSearchCV, `refit=\"mae\"`). The operational goal is minimizing the "
        "average number of meals mispredicted. Validation RMSE is scored on the same folds.",
        "- **The final model is chosen by lowest TimeSeriesSplit validation MAE** "
        "(validation RMSE as tiebreak) across all four candidates. Linear Regression, "
        "which has no tuning grid, is cross-validated on the same folds with the same scorers.",
        "- **Test-period metrics are computed only after selection.** The selected model is "
        "evaluated once on the untouched test period; test metrics for the other models are "
        "reported below for research comparison and did not influence tuning or selection.",
        "",
        "## Search Method Per Model",
        "",
        "- **Decision Tree** — exhaustive `GridSearchCV` (240 combinations × 5 folds = "
        "1200 fits). Cheap because a single shallow tree fits almost instantly.",
        "- **Random Forest** — `RandomizedSearchCV` (25 of 324 possible combinations). "
        "The full grid (up to 300 trees × 324 combinations × 5 folds) was too slow for "
        "a development machine; randomized sampling covers the space at a fraction of the cost.",
        "- **XGBoost** — `RandomizedSearchCV` (40 of 4,374 possible combinations, fixed "
        "seed 42). The full grid would mean ~21,870 fits — impractical here; randomized "
        "search still explores the depth/learning-rate/regularization combinations "
        "meant to curb the overfitting seen in Phase 4.",
        "- **Linear Regression** — not tuned (no hyperparameters meaningfully affect "
        "ordinary least squares here). It is a candidate ML model, not a naive baseline, "
        "and receives a TimeSeriesSplit validation score on the same folds as the others.",
        "",
        "## Best Parameters Found",
        "",
    ]
    for name, params in best_params_all.items():
        lines.append(f"**{PRETTY_NAMES[name]}:**")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(params, indent=2, default=str))
        lines.append("```")
        lines.append("")

    lines += [
        "## Tuned Results (Test Set — research comparison, not used for selection)",
        "",
        tuned_metrics_df.drop(columns=["Best CV MAE"], errors="ignore").to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Default-Hyperparameter vs Tuned Comparison (test set, research only)",
        "",
        comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]

    lines.append("## Did Tuning Reduce Overfitting?")
    lines.append("")
    for _, row in comparison_df.iterrows():
        verdict = (
            "improved" if row["MAE Improvement"] > 0 else
            "did not improve (tuned model performed worse)" if row["MAE Improvement"] < 0 else
            "was unchanged"
        )
        lines.append(
            f"- **{row['Model']}**: Test MAE {verdict} by {row['MAE Improvement']:.2f} meals "
            f"({row['Baseline Overfitting Status']} → {row['Tuned Overfitting Status']})."
        )
    lines.append("")

    validation_rows = [
        "| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |",
        "|---|---|---|---|---|---|",
    ]
    for rank, score in enumerate(ranked, start=1):
        rmse = f"{score.cv_rmse:.3f}" if score.cv_rmse is not None else "n/a"
        validation_rows.append(
            f"| {rank} | {score.model} | {score.cv_mae:.3f} | {score.cv_mae_std:.3f} | {rmse} | {score.n_folds} |"
        )

    lines += [
        "## Final Model Selection (validation only)",
        "",
        f"Selection criterion: **{SELECTION_CRITERION}**, with {TIEBREAK_CRITERION} as the "
        "tiebreak. Scores are means over the TimeSeriesSplit validation folds of the "
        "training period. The test period was not used to rank or select models.",
        "",
        *validation_rows,
        "",
        f"**Selected final model: {final_best['Model']}** "
        f"(validation MAE={final_best['Validation MAE']:.3f}).",
        "",
        "## Final Test Evaluation (selected model, evaluated once)",
        "",
        f"- Test MAE: **{final_best['Test MAE']:.3f}** meals",
        f"- Test RMSE: **{final_best['Test RMSE']:.3f}** meals",
        f"- Test R²: **{final_best['Test R2']:.3f}**",
        f"- Train/test generalization: {final_best['Generalization Gap Status']}",
        "",
        "## Naive Forecasting Baselines",
        "",
        "Two simple rules, not fit or tuned, evaluated on the same test period as the ML models. "
        "Both use only each school's own realised meal demand from strictly earlier observations:",
        "",
        "- **Persistence / Last Observation** — prediction(T) = meals served on the school's "
        "most recent available observation before T.",
        "- **7-Day Historical Mean** — prediction(T) = mean meals served over the school's "
        "previous 7 available observations (T excluded: shift by one before the rolling window).",
        "",
        "Linear Regression is an ML candidate, not a naive baseline. `projected_students` is not "
        "used as a baseline.",
        "",
        "### Methods Compared",
        "",
        "Validation columns are empty for the naive baselines: they were not evaluated on the "
        "validation folds. Test columns for non-selected ML models are for research comparison only.",
        "",
        model_comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Selected Model vs Naive Baselines (test period)",
        "",
        naive_comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    for _, row in naive_comparison_df.iterrows():
        pct = row["MAE Improvement (%)"]
        pct_text = f" ({pct:.1f}%)" if pct is not None and pd.notna(pct) else ""
        if row["MAE Improvement"] > 0:
            verdict = f"beats it by {row['MAE Improvement']:.2f} meals MAE{pct_text}"
        elif row["MAE Improvement"] < 0:
            verdict = f"does NOT beat it: MAE is worse by {-row['MAE Improvement']:.2f} meals{pct_text}"
        else:
            verdict = "ties it on MAE"
        lines.append(f"- vs **{row['Naive Baseline']}**: {row['Selected Model']} {verdict}.")
    if not settings.USE_GOOGLE_SHEET:
        lines += [
            "",
            "## Limitations of Synthetic Data",
            "",
            "All tuning and evaluation above used the synthetic dataset generated for "
            "development (`src/synthetic_data.py`). Relationships between rainfall, "
            "holidays, exams, and attendance were hand-designed to be plausible, not "
            "fit to any real county's records. Hyperparameters selected here are only "
            "known to be good for this synthetic data's noise structure and scale.",
            "",
            "## What to Repeat With Real Data",
            "",
            "1. Re-run `run_pipeline.py` and `run_eda.py` against the real dataset once "
            "available (point `BITETRACKER_DATA_PATH`/`ACTIVE_DATA_PATH` at it).",
            "2. Re-run `run_training.py` for fresh baselines — real data's noise level "
            "and feature relationships will differ from the synthetic generator's assumptions.",
            "3. Re-run this tuning phase (`run_tuning.py`) from scratch — do not reuse "
            "the hyperparameters found here; they were fit to synthetic patterns.",
            "4. Re-check the TimeSeriesSplit fold sizes — a real dataset with different "
            "row counts or a longer history may warrant more or fewer folds.",
            "",
        ]

    out_path = settings.REPORTS_DIR / "Hyperparameter_Tuning_Report.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
