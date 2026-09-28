"""One-command BiteTracker ML pipeline.

Runs the same stages as `run_pipeline.py` -> `run_training.py` -> `run_tuning.py`,
in one process, by calling their functions (no ML logic is duplicated here):

    [1/7] load + clean the dataset
    [2/7] engineer features, chronological 80/20 split
    [3/7] train the four baseline models (default hyperparameters)
    [4/7] tune Decision Tree / Random Forest / XGBoost (TimeSeriesSplit, training data only)
    [5/7] evaluate the tuned models on the held-out test set
    [6/7] select the final model (lowest test MAE, then RMSE, then R²)
    [7/7] save models, metrics, metadata, charts and reports

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --data-path path/to/dataset.csv

Without --data-path, the dataset comes from `config.settings.ACTIVE_DATA_PATH`
(the BITETRACKER_DATA_PATH environment variable, or the synthetic sample by default).

Any failure stops the run with a non-zero exit status, so CI can detect it.
"""

from __future__ import annotations

import argparse
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
from src.feature_engineering import engineer_features  # noqa: E402
from src.model_training import prepare_features_and_target  # noqa: E402
from src.pipeline import save_processed_dataset  # noqa: E402
from src.preprocessing import run_preprocessing  # noqa: E402

TOTAL_STAGES = 7


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BiteTracker ML pipeline with one command.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Input CSV with the BiteTracker schema. Defaults to BITETRACKER_DATA_PATH, "
        "or the synthetic sample dataset if that is not set.",
    )
    return parser.parse_args(argv)


def resolve_data_path(data_path: Path | None) -> Path:
    """Use --data-path if given, otherwise the configured ACTIVE_DATA_PATH."""
    if data_path is None:
        return settings.ACTIVE_DATA_PATH
    return data_path.expanduser().resolve()


class _Progress:
    """Prints '[n/7] ...' stage headers and remembers the current stage for error messages."""

    def __init__(self) -> None:
        self.current = "startup"

    def stage(self, number: int, message: str) -> None:
        self.current = f"[{number}/{TOTAL_STAGES}] {message}"
        print(f"\n{self.current}", flush=True)


def run(data_path: Path, progress: _Progress) -> dict:
    """Run every stage in order and return a summary of the result."""
    progress.stage(1, "Preparing data (load + clean)...")
    raw_df = load_dataset(path=data_path)
    clean_df = run_preprocessing(raw_df)
    print(f"      {len(raw_df)} raw rows -> {len(clean_df)} cleaned rows")

    progress.stage(2, "Engineering features and splitting chronologically...")
    processed_df = engineer_features(clean_df)
    save_processed_dataset(processed_df)
    train_df, test_df = run_training.split_train_test(processed_df)
    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(test_df)
    print(
        f"      train {len(train_df)} rows ({train_df[settings.DATE_COLUMN].min().date()} to "
        f"{train_df[settings.DATE_COLUMN].max().date()}), test {len(test_df)} rows "
        f"({test_df[settings.DATE_COLUMN].min().date()} to {test_df[settings.DATE_COLUMN].max().date()})"
    )

    progress.stage(3, "Training baseline models (Linear Regression, Decision Tree, Random Forest, XGBoost)...")
    baseline_metrics_df, baseline_predictions, baseline_models = run_training.train_and_evaluate_baselines(
        train_df, test_df
    )
    baseline_df = baseline_metrics_df.set_index("Model")

    progress.stage(4, f"Tuning models (TimeSeriesSplit, {settings.CV_N_SPLITS} folds, training data only)...")
    searches, fold_info = run_tuning.tune_models(X_train, y_train, settings.CV_N_SPLITS)

    progress.stage(5, "Evaluating tuned models on the held-out test set...")
    evaluation = run_tuning.evaluate_tuned_models(searches, X_train, y_train, X_test, y_test, test_df, baseline_df)

    progress.stage(6, "Selecting final model (lowest test MAE, then RMSE, then R²)...")
    final_best, ranked = run_tuning.select_final_model(evaluation["tuned_metrics_rows"], baseline_df)
    print(f"      ranking: {', '.join(r['Model'] for r in ranked)}")

    progress.stage(7, "Saving artifacts...")
    # Baseline models first: the final model is copied from them if Linear Regression wins.
    run_training.save_baseline_outputs(baseline_metrics_df, baseline_predictions, baseline_models, train_df, test_df)
    run_tuning.save_tuning_outputs(
        evaluation, fold_info, final_best, ranked, train_df, test_df, X_test, y_test, settings.CV_N_SPLITS
    )

    return {"final_best": final_best, "ranked": ranked, "baseline_metrics": baseline_metrics_df}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = resolve_data_path(args.data_path)
    is_synthetic = data_path.resolve() == settings.SYNTHETIC_DATA_PATH.resolve()

    print("BiteTracker one-command ML pipeline")
    print(f"Dataset: {data_path}")
    if is_synthetic:
        print("Data source: SYNTHETIC development dataset (not real school records).")
    else:
        print(
            "WARNING: non-default dataset. The generated reports, model metadata and app badges "
            "still describe the data as synthetic; update them before presenting results."
        )

    progress = _Progress()
    start = time.perf_counter()
    try:
        summary = run(data_path, progress)
    except Exception:
        print(f"\nPIPELINE FAILED during {progress.current}", file=sys.stderr, flush=True)
        raise

    final_best = summary["final_best"]
    print(f"\nPipeline finished in {time.perf_counter() - start:.0f}s.")
    print(
        f"Final model: {final_best['Model']} | Test MAE {final_best['Test MAE']:.2f} | "
        f"Test RMSE {final_best['Test RMSE']:.2f} | Test R² {final_best['Test R2']:.3f}"
    )
    print(f"Saved to: {settings.MODELS_DIR / 'final_best_model.joblib'}")
    if is_synthetic:
        print("These metrics come from synthetic data and say nothing about real-world accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
