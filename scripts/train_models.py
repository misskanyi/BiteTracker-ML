"""One-command BiteTracker ML pipeline.

Runs the same stages as `run_pipeline.py` -> `run_training.py` -> `run_tuning.py`,
in one process, by calling their functions (no ML logic is duplicated here):

    [1/9] load the dataset
    [2/9] preprocess (stateless cleaning)
    [3/9] fit imputation/outlier statistics on the training period only, apply them
          to every row, and engineer features
    [4/9] chronological 80/20 train/test split
    [5/9] evaluate the naive forecasting baselines (persistence, 7-day mean) on the test period
    [6/9] train the four ML candidates: default-hyperparameter fits, TimeSeriesSplit
          tuning of Decision Tree / Random Forest / XGBoost, and TimeSeriesSplit
          cross-validation of Linear Regression (training period only)
    [7/9] select the final model by lowest TimeSeriesSplit validation MAE
          (validation RMSE tiebreak) — no test-period data involved
    [8/9] evaluate the selected model once on the held-out test period
    [9/9] save models, metrics, metadata, charts and reports

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --data-path path/to/dataset.csv

Without --data-path, the dataset is the Food4Education Google Sheet when
BITETRACKER_GOOGLE_SHEET_URL is set (read live from the link, never saved),
otherwise the file named by the BITETRACKER_DATA_PATH environment variable.
If neither is set the run stops: real data is the default, so the synthetic
sample is only used when asked for explicitly (CI does this, as it has no
access to the real sheet):

    python scripts/train_models.py --data-path data/sample/synthetic_school_meals.csv

Any failure stops the run with a non-zero exit status, so CI can detect it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_training  # noqa: E402
import run_tuning  # noqa: E402
from config import settings  # noqa: E402
from src.data_loader import load_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    FOOD4EDUCATION_SOURCE,
    SYNTHETIC_SOURCE,
    feature_columns_for,
)
from src.google_sheet_loader import TARGET_SOURCE_COLUMN  # noqa: E402
from src.model_training import prepare_features_and_target  # noqa: E402
from src.naive_baselines import evaluate_naive_baselines  # noqa: E402
from src.pipeline import build_training_data, save_processed_dataset  # noqa: E402
from src.preprocessing import clean_raw_data  # noqa: E402

TOTAL_STAGES = 9


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BiteTracker ML pipeline with one command.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Input CSV with the BiteTracker schema. Defaults to the Food4Education Google Sheet "
        "(BITETRACKER_GOOGLE_SHEET_URL), then BITETRACKER_DATA_PATH.",
    )
    return parser.parse_args(argv)


def resolve_data_path(data_path: Path | None) -> Path | None:
    """Use --data-path if given, otherwise the configured dataset.

    Returns None when the configured dataset is the Google Sheet, which
    `load_dataset(path=None)` then reads from its link.
    """
    if data_path is None:
        return None if settings.USE_GOOGLE_SHEET else settings.ACTIVE_DATA_PATH
    return data_path.expanduser().resolve()


class _Progress:
    """Prints '[n/9] ...' stage headers and remembers the current stage for error messages."""

    def __init__(self) -> None:
        self.current = "startup"

    def stage(self, number: int, message: str) -> None:
        self.current = f"[{number}/{TOTAL_STAGES}] {message}"
        print(f"\n{self.current}", flush=True)


def run(data_path: Path | None, progress: _Progress, data_source: str = SYNTHETIC_SOURCE) -> dict:
    """Run every stage in order and return a summary of the result."""
    feature_columns = feature_columns_for(data_source)

    progress.stage(1, "Loading data...")
    raw_df = load_dataset(path=data_path)
    print(f"      {len(raw_df)} raw rows")

    progress.stage(2, "Preprocessing (cleaning)...")
    cleaned_df = clean_raw_data(raw_df)
    print(f"      {len(cleaned_df)} cleaned rows")

    progress.stage(3, "Fitting preprocessing on the training period and engineering features...")
    data = build_training_data(cleaned_df)
    clean_df, processed_df = data.clean_df, data.processed_df
    print(f"      imputation/outlier statistics fitted on {data.preprocessing.fit_rows} rows up to {data.cutoff_date.date()}")
    if data.test_targets_capped:
        print(f"      note: {data.test_targets_capped} test-period target values were capped by the training-period fences")
    save_processed_dataset(processed_df)

    progress.stage(4, "Splitting chronologically (train/test)...")
    train_df, test_df = data.train_df, data.test_df
    X_train, y_train = prepare_features_and_target(train_df, feature_columns)
    X_test, y_test = prepare_features_and_target(test_df, feature_columns)
    print(
        f"      train {len(train_df)} rows ({train_df[settings.DATE_COLUMN].min().date()} to "
        f"{train_df[settings.DATE_COLUMN].max().date()}), test {len(test_df)} rows "
        f"({test_df[settings.DATE_COLUMN].min().date()} to {test_df[settings.DATE_COLUMN].max().date()})"
    )

    progress.stage(5, "Evaluating naive baselines (Persistence, 7-Day Historical Mean) on the test period...")
    # History = every cleaned observation, so forecasts only ever look backwards from each test date.
    naive_metrics_df, naive_predictions = evaluate_naive_baselines(clean_df, test_df)
    for _, row in naive_metrics_df.iterrows():
        print(f"      {row['Method']}: test MAE {row['Test MAE']:.2f}, RMSE {row['Test RMSE']:.2f}")

    progress.stage(
        6,
        f"Training ML candidates (default fits; TimeSeriesSplit tuning/CV, {settings.CV_N_SPLITS} folds, "
        "training period only)...",
    )
    baseline_metrics_df, baseline_predictions, baseline_models = run_training.train_and_evaluate_baselines(
        train_df, test_df, feature_columns
    )
    baseline_df = baseline_metrics_df.set_index("Model")
    searches, fold_info = run_tuning.tune_models(X_train, y_train, settings.CV_N_SPLITS)
    validation_scores = run_tuning.collect_validation_scores(searches, X_train, y_train, settings.CV_N_SPLITS)

    progress.stage(7, "Selecting final model (lowest TimeSeriesSplit validation MAE)...")
    selected, ranked = run_tuning.select_final_model(validation_scores)
    for rank, score in enumerate(ranked, start=1):
        print(f"      {rank}. {score.model}: validation MAE {score.cv_mae:.3f}")
    print(f"      selected: {selected.model}")

    progress.stage(8, f"Evaluating selected model ({selected.model}) on the held-out test period...")
    final_model, model_kind = run_tuning.fit_selected_model(selected, searches, X_train, y_train)
    final_best = run_tuning.evaluate_selected_model(
        selected, final_model, model_kind, X_train, y_train, X_test, y_test
    )
    # Research comparison tables only: every tuned model on the test period, after selection.
    evaluation = run_tuning.evaluate_tuned_models(searches, X_train, y_train, X_test, y_test, test_df, baseline_df)

    progress.stage(9, "Saving artifacts...")
    run_training.save_baseline_outputs(
        baseline_metrics_df, baseline_predictions, baseline_models, train_df, test_df, feature_columns
    )
    run_tuning.save_tuning_outputs(
        evaluation, fold_info, final_best, final_model, ranked, naive_metrics_df, naive_predictions,
        baseline_df, train_df, test_df, X_test, y_test, settings.CV_N_SPLITS,
        extra_metadata=run_metadata(data_source, data),
    )

    return {
        "final_best": final_best,
        "ranked": ranked,
        "baseline_metrics": baseline_metrics_df,
        "naive_metrics": naive_metrics_df,
    }


def run_metadata(data_source: str, data) -> dict:
    """Data source, target definition and fitted preprocessing, for the model metadata."""
    return {
        "data_source": data_source,
        "target_source_column": TARGET_SOURCE_COLUMN if data_source == FOOD4EDUCATION_SOURCE else "meals_served",
        "preprocessing": {
            "fitted_on": "training period only",
            "test_targets_capped": data.test_targets_capped,
            **data.preprocessing.to_dict(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.data_path is None and not settings.USE_GOOGLE_SHEET and not os.getenv("BITETRACKER_DATA_PATH"):
        print(
            "No real dataset configured: set BITETRACKER_GOOGLE_SHEET_URL in .env to the Food4Education "
            "Google Sheet link. To train on the synthetic sample instead, ask for it explicitly with "
            "--data-path data/sample/synthetic_school_meals.csv",
            file=sys.stderr,
        )
        return 2
    if args.data_path is not None:
        # An explicit file overrides the Google Sheet, including the results label.
        settings.USE_GOOGLE_SHEET = False
    data_path = resolve_data_path(args.data_path)
    is_synthetic = data_path is not None and data_path.resolve() == settings.SYNTHETIC_DATA_PATH.resolve()

    print("BiteTracker one-command ML pipeline")
    print(f"Dataset: {data_path or settings.GOOGLE_SHEET_DATASET_NAME}")
    if is_synthetic:
        print("Data source: SYNTHETIC development dataset (not real school records).")
    elif data_path is None:
        print("Data source: Food4Education Google Sheet, read live from its link (not saved to disk).")
    else:
        print(
            "WARNING: non-default dataset. The generated reports, model metadata and app badges "
            "still describe the data as synthetic; update them before presenting results."
        )

    progress = _Progress()
    start = time.perf_counter()
    try:
        summary = run(data_path, progress, FOOD4EDUCATION_SOURCE if data_path is None else SYNTHETIC_SOURCE)
    except Exception:
        print(f"\nPIPELINE FAILED during {progress.current}", file=sys.stderr, flush=True)
        raise

    final_best = summary["final_best"]
    print(f"\nPipeline finished in {time.perf_counter() - start:.0f}s.")
    if is_synthetic:
        print(f"=== {run_tuning.SYNTHETIC_LABEL} ===")
    print(
        f"Final model: {final_best['Model']} (selected by {run_tuning.SELECTION_CRITERION}: "
        f"{final_best['Validation MAE']:.2f})"
    )
    print(
        f"Final test: MAE {final_best['Test MAE']:.2f} | "
        f"RMSE {final_best['Test RMSE']:.2f} | R² {final_best['Test R2']:.3f}"
    )
    for _, row in summary["naive_metrics"].iterrows():
        diff = row["Test MAE"] - final_best["Test MAE"]
        verdict = "beats" if diff > 0 else "does NOT beat" if diff < 0 else "ties"
        print(f"  {final_best['Model']} {verdict} {row['Method']} (test MAE {row['Test MAE']:.2f}, diff {diff:+.2f})")
    print(f"Saved to: {settings.MODELS_DIR / 'final_best_model.joblib'}")
    if is_synthetic:
        print("These metrics come from synthetic data and say nothing about real-world accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
