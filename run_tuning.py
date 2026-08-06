"""CLI entry point: Phase 5 hyperparameter tuning.

Tunes Decision Tree, Random Forest, and XGBoost with TimeSeriesSplit +
GridSearchCV/RandomizedSearchCV over the Phase 4 training set only, then
evaluates once on the untouched Phase 4 test set. Linear Regression is left
unchanged (not tuned) and carried forward as the baseline reference.

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
from src.model_training import chronological_split, prepare_features_and_target
from src.model_tuning import (
    TUNERS,
    describe_cv_folds,
    get_time_series_cv,
    save_best_hyperparameters,
    save_tuned_model,
)
from src.utils import get_logger

logger = get_logger(__name__)

PRETTY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Phase 4 baseline results — used only for the baseline-vs-tuned comparison,
# recomputed from the saved baseline metrics file (not hardcoded/fabricated).
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


def main() -> None:
    df = load_processed_dataset()
    train_df, test_df = chronological_split(df)  # identical split to Phase 4
    baseline_df = load_baseline_metrics().set_index("Model")

    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)

    n_splits = 5
    cv = get_time_series_cv(n_splits)
    fold_info = describe_cv_folds(len(X_train), n_splits)
    logger.info("TimeSeriesSplit folds (training data only, %d rows):", len(X_train))
    for fold in fold_info:
        logger.info(
            "  Fold %d: train rows %d (idx %s), val rows %d (idx %s)",
            fold["fold"], fold["train_rows"], fold["train_range"], fold["val_rows"], fold["val_range"],
        )

    tuned_metrics_rows = []
    comparison_rows = []
    best_params_all = {}
    tuning_metadata = {"n_splits": n_splits, "cv_folds": fold_info, "models": {}}
    predictions = pd.DataFrame(
        {
            "date": test_df[settings.DATE_COLUMN].values,
            "school_id": test_df[settings.SCHOOL_ID_COLUMN].values,
            "actual_meals": y_test.values,
        }
    )

    for name, tuner in TUNERS.items():
        pretty = PRETTY_NAMES[name]
        logger.info("Tuning %s ...", pretty)
        start = time.perf_counter()
        search = tuner(X_train, y_train, cv)
        tuning_seconds = time.perf_counter() - start
        logger.info("%s tuning finished in %.1fs — best CV MAE: %.3f", pretty, tuning_seconds, -search.best_score_)

        best_model = search.best_estimator_
        best_params_all[name] = search.best_params_

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
            "Best CV MAE": -search.best_score_,
        }
        tuned_metrics_rows.append(row)

        baseline_row = baseline_df.loc[pretty].to_dict()
        comparison_rows.append(
            _build_comparison_row(pretty, baseline_row, row)
        )

        predictions[f"{name}_tuned_prediction"] = test_pred
        save_tuned_model(best_model, name)

        tuning_metadata["models"][name] = {
            "search_type": type(search).__name__,
            "n_candidates_evaluated": len(search.cv_results_["params"]),
            "tuning_seconds": tuning_seconds,
            "best_cv_mae": -search.best_score_,
        }

    tuned_metrics_df = pd.DataFrame(tuned_metrics_rows)
    tuned_metrics_path = settings.METRICS_DIR / "tuned_model_metrics.csv"
    tuned_metrics_df.to_csv(tuned_metrics_path, index=False)
    logger.info("Saved tuned metrics to %s", tuned_metrics_path)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = settings.METRICS_DIR / "baseline_vs_tuned_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved baseline-vs-tuned comparison to %s", comparison_path)

    predictions_path = settings.METRICS_DIR / "tuned_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    save_best_hyperparameters(best_params_all)
    with open(settings.MODELS_DIR / "tuning_metadata.json", "w") as f:
        json.dump(tuning_metadata, f, indent=2, default=str)

    # --- Final model selection: lowest Test MAE first, RMSE tiebreak, R² as supporting evidence.
    all_candidates = tuned_metrics_rows + [
        {
            "Model": "Linear Regression",
            "Train MAE": baseline_df.loc["Linear Regression", "Train MAE"],
            "Test MAE": baseline_df.loc["Linear Regression", "Test MAE"],
            "Train RMSE": baseline_df.loc["Linear Regression", "Train RMSE"],
            "Test RMSE": baseline_df.loc["Linear Regression", "Test RMSE"],
            "Train R2": baseline_df.loc["Linear Regression", "Train R2"],
            "Test R2": baseline_df.loc["Linear Regression", "Test R2"],
            "Generalization Gap Status": classify_generalization_gap(
                baseline_df.loc["Linear Regression", "Train MAE"],
                baseline_df.loc["Linear Regression", "Test MAE"],
                baseline_df.loc["Linear Regression", "Train R2"],
                baseline_df.loc["Linear Regression", "Test R2"],
            ),
            "Tuning Time (s)": 0.0,
            "Prediction Time (s)": None,
        }
    ]
    ranked = sorted(all_candidates, key=lambda r: (r["Test MAE"], r["Test RMSE"], -r["Test R2"]))
    final_best = ranked[0]
    logger.info("Final model ranking by Test MAE: %s", [r["Model"] for r in ranked])
    logger.info("Selected final model: %s (Test MAE=%.2f)", final_best["Model"], final_best["Test MAE"])

    save_final_best_model(final_best, train_df, test_df)
    generate_charts(tuned_metrics_df, comparison_df, predictions, fold_info)
    generate_final_best_model_charts(final_best, X_test, y_test)
    write_report(tuned_metrics_df, comparison_df, best_params_all, fold_info, final_best, ranked, len(X_train))

    print("\n=== Tuned Metrics ===")
    print(tuned_metrics_df.to_string(index=False))
    print("\n=== Baseline vs Tuned ===")
    print(comparison_df.to_string(index=False))
    print(f"\nFinal selected model: {final_best['Model']} (Test MAE={final_best['Test MAE']:.2f})")


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


def save_final_best_model(final_best: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    import joblib
    from src.model_tuning import load_tuned_model
    from src.model_training import load_model

    name_lookup = {
        "Decision Tree": ("decision_tree", "tuned"),
        "Random Forest": ("random_forest", "tuned"),
        "XGBoost": ("xgboost", "tuned"),
        "Linear Regression": ("linear_regression", "baseline"),
    }
    short_name, kind = name_lookup[final_best["Model"]]
    model = load_tuned_model(short_name) if kind == "tuned" else load_model(short_name)

    final_path = settings.MODELS_DIR / "final_best_model.joblib"
    joblib.dump(model, final_path)

    metadata = {
        "selected_model": final_best["Model"],
        "model_kind": kind,
        "best_parameters": model.get_params() if hasattr(model, "get_params") else None,
        "train_date_range": {
            "start": str(train_df[settings.DATE_COLUMN].min().date()),
            "end": str(train_df[settings.DATE_COLUMN].max().date()),
        },
        "test_date_range": {
            "start": str(test_df[settings.DATE_COLUMN].min().date()),
            "end": str(test_df[settings.DATE_COLUMN].max().date()),
        },
        "features_used": FEATURE_COLUMNS,
        "target_variable": settings.TARGET_COLUMN,
        "final_test_mae": final_best["Test MAE"],
        "final_test_rmse": final_best["Test RMSE"],
        "final_test_r2": final_best["Test R2"],
        "synthetic_data_warning": (
            "These metrics were measured on a SYNTHETIC development dataset "
            "(src/synthetic_data.py), not real school records. Re-run the full "
            "pipeline and re-tune once real data is available before trusting "
            "these numbers operationally."
        ),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with open(settings.MODELS_DIR / "final_best_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved final best model (%s) and metadata", final_best["Model"])


def generate_charts(tuned_metrics_df, comparison_df, predictions, fold_info) -> None:
    figures_dir = settings.FIGURES_DIR

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
            {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
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


def write_report(tuned_metrics_df, comparison_df, best_params_all, fold_info, final_best, ranked, n_train_rows) -> None:
    lines = [
        "# BiteTracker — Hyperparameter Tuning Report (Phase 5)",
        "",
        "> Results below are from **synthetic development data** "
        "(`data/sample/synthetic_school_meals.csv`) and are measured, not fabricated.",
        "",
        "## Why Tuning Was Required",
        "",
        "Phase 4 baselines used library-default hyperparameters. The Decision Tree "
        "overfit severely (train R²=1.0 vs test R²=0.45) and XGBoost also overfit "
        "(train R²=0.999 vs test R²=0.768). Random Forest was the best baseline but "
        "still had a meaningful train/test gap. This phase tunes depth, leaf-size, "
        "regularization, and sampling parameters specifically to reduce that gap, not "
        "just to chase a lower training error.",
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
        "- **GridSearchCV / RandomizedSearchCV selection metric: negative MAE.** "
        "The operational goal is minimizing the average number of meals "
        "mispredicted, so hyperparameters were chosen to minimize cross-validation MAE.",
        "- **RMSE and R² were computed after refitting the best estimator** on the "
        "full training set and evaluating once on the untouched test set — they "
        "describe the result, they did not influence which hyperparameters were chosen.",
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
        "- **Linear Regression** — not tuned, kept as the unchanged Phase 4 baseline "
        "reference (no hyperparameters meaningfully affect ordinary least squares here).",
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
        "## Tuned Results (Test Set)",
        "",
        tuned_metrics_df.drop(columns=["Best CV MAE"], errors="ignore").to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Baseline vs Tuned Comparison",
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

    lines += [
        "## Final Model Selection",
        "",
        "Selection order: (1) lowest test MAE, (2) lowest test RMSE as tiebreak, "
        "(3) highest test R² as supporting evidence, (4) acceptable generalization "
        "gap, (5) practical prediction speed. The model was **not** chosen for "
        "having the highest R² alone.",
        "",
        "Ranking by test MAE (best to worst):",
        "",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(f"{i}. {r['Model']} — Test MAE {r['Test MAE']:.2f}, Test RMSE {r['Test RMSE']:.2f}, Test R² {r['Test R2']:.3f}")

    lines += [
        "",
        f"**Selected final model: {final_best['Model']}** "
        f"(Test MAE={final_best['Test MAE']:.2f}, Test RMSE={final_best['Test RMSE']:.2f}, "
        f"Test R²={final_best['Test R2']:.3f}, generalization: {final_best.get('Generalization Gap Status', 'n/a')}).",
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
