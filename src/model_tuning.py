"""Hyperparameter tuning for Decision Tree, Random Forest, and XGBoost.

Linear Regression is intentionally NOT tuned (no hyperparameters meaningfully
affect ordinary least squares here); it is still a candidate model and gets a
TimeSeriesSplit validation score on the same folds — see src/model_selection.py.

Cross-validation strategy: `TimeSeriesSplit` over the Phase 4 *training*
portion only. The final 20% test set from Phase 4 is never touched by
GridSearchCV/RandomizedSearchCV — it is reserved for one unbiased evaluation
after the best hyperparameters have been selected.

Search method per model:
- Decision Tree: full `GridSearchCV` — the grid (5*4*4*3=240 combos) is
  cheap because a single shallow tree fits almost instantly, so an
  exhaustive search over 240*5=1200 fits stays fast.
- Random Forest: `RandomizedSearchCV` (n_iter=25) — the full grid
  (3*4*3*3*3=324 combos, up to 300 trees each) would mean over 1600 fits of
  a comparatively expensive ensemble; randomized search samples the same
  space at a fraction of the runtime while still covering it broadly.
- XGBoost: `RandomizedSearchCV` (n_iter=40) — the full grid has 4374
  combinations; at 5 folds that's ~21,870 fits, impractical for a
  development machine. Randomized search with a fixed seed keeps runtime
  practical while still exploring depth/learning-rate/regularization
  combinations meant to curb the overfitting seen in the Phase 4 baseline.

Scoring: every search scores both MAE and RMSE on each validation fold
(CV_SCORING) and picks hyperparameters by MAE (refit="mae"; operationally,
minimizing average meals mispredicted matters most). The validation RMSE is
kept only as the tiebreak for final model selection. Test-set metrics are
computed after selection and never feed back into tuning or selection.
"""

# In simple terms: tries out many settings ("knobs") for the Decision Tree,
# Random Forest, and XGBoost models to find the version of each that
# predicts best.

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import settings
from src.model_selection import CV_REFIT_METRIC, CV_SCORING
from src.utils import get_logger

logger = get_logger(__name__)

TUNED_MODEL_FILENAMES = {
    "decision_tree": "decision_tree_tuned.joblib",
    "random_forest": "random_forest_tuned.joblib",
    "xgboost": "xgboost_tuned.joblib",
}

DECISION_TREE_PARAM_GRID = {
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "sqrt", "log2"],
}

RANDOM_FOREST_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.7],
    "bootstrap": [True],
}

XGBOOST_PARAM_DIST = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 5, 10],
}

RANDOM_FOREST_N_ITER = 25
XGBOOST_N_ITER = 40


def get_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return a chronology-preserving TimeSeriesSplit (no shuffling)."""
    return TimeSeriesSplit(n_splits=n_splits)


def describe_cv_folds(n_rows: int, n_splits: int = 5) -> list[dict]:
    """Return the row-count ranges each TimeSeriesSplit fold will use.

    Purely informational (for documentation/logging) — mirrors sklearn's own
    fold-boundary logic so reports can state exact fold sizes without
    actually running the split.
    """
    cv = get_time_series_cv(n_splits)
    dummy_X = pd.DataFrame({"x": range(n_rows)})
    folds = []
    for i, (train_idx, val_idx) in enumerate(cv.split(dummy_X), start=1):
        folds.append(
            {
                "fold": i,
                "train_rows": len(train_idx),
                "train_range": (int(train_idx.min()), int(train_idx.max())),
                "val_rows": len(val_idx),
                "val_range": (int(val_idx.min()), int(val_idx.max())),
            }
        )
    return folds


def tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit) -> GridSearchCV:
    """Exhaustive GridSearchCV over the decision tree parameter grid."""
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=settings.RANDOM_SEED),
        DECISION_TREE_PARAM_GRID,
        scoring=CV_SCORING,
        refit=CV_REFIT_METRIC,
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit, n_iter: int = RANDOM_FOREST_N_ITER
) -> RandomizedSearchCV:
    """RandomizedSearchCV over the random forest parameter space (see module docstring for why)."""
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=settings.RANDOM_SEED, n_jobs=1),
        RANDOM_FOREST_PARAM_DIST,
        n_iter=n_iter,
        scoring=CV_SCORING,
        refit=CV_REFIT_METRIC,
        cv=cv,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit, n_iter: int = XGBOOST_N_ITER
) -> RandomizedSearchCV:
    """RandomizedSearchCV over the XGBoost parameter space (see module docstring for why)."""
    search = RandomizedSearchCV(
        XGBRegressor(random_state=settings.RANDOM_SEED, n_jobs=1, verbosity=0),
        XGBOOST_PARAM_DIST,
        n_iter=n_iter,
        scoring=CV_SCORING,
        refit=CV_REFIT_METRIC,
        cv=cv,
        random_state=settings.RANDOM_SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


TUNERS = {
    "decision_tree": tune_decision_tree,
    "random_forest": tune_random_forest,
    "xgboost": tune_xgboost,
}


def save_tuned_model(model, name: str, models_dir: Path | None = None) -> Path:
    """Persist a tuned model to models/<name>_tuned.joblib."""
    out_dir = models_dir or settings.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TUNED_MODEL_FILENAMES[name]
    joblib.dump(model, path)
    logger.info("Saved tuned %s to %s", name, path)
    return path


def load_tuned_model(name: str, models_dir: Path | None = None):
    """Load a previously saved tuned model by its short name."""
    models_dir = models_dir or settings.MODELS_DIR
    path = models_dir / TUNED_MODEL_FILENAMES[name]
    if not path.exists():
        raise FileNotFoundError(f"No tuned model found at {path}")
    return joblib.load(path)


def save_best_hyperparameters(best_params: dict[str, dict], path: Path | None = None) -> Path:
    """Persist the best hyperparameters found for every tuned model to JSON."""
    out_path = path or (settings.MODELS_DIR / "best_hyperparameters.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    logger.info("Saved best hyperparameters to %s", out_path)
    return out_path


def load_best_hyperparameters(path: Path | None = None) -> dict:
    in_path = path or (settings.MODELS_DIR / "best_hyperparameters.json")
    with open(in_path) as f:
        return json.load(f)


def compare_baseline_vs_tuned(model_name: str, baseline_row: dict, tuned_row: dict) -> dict:
    """Build one row of the baseline-vs-tuned comparison table.

    Improvements are baseline-minus-tuned (positive = tuned is better,
    negative = tuned performed worse — never masked as an improvement).
    """
    from src.model_evaluation import classify_generalization_gap

    mae_improvement = baseline_row["Test MAE"] - tuned_row["Test MAE"]
    rmse_improvement = baseline_row["Test RMSE"] - tuned_row["Test RMSE"]
    r2_improvement = tuned_row["Test R2"] - baseline_row["Test R2"]

    baseline_status = classify_generalization_gap(
        baseline_row["Train MAE"], baseline_row["Test MAE"], baseline_row["Train R2"], baseline_row["Test R2"]
    )
    tuned_status = classify_generalization_gap(
        tuned_row["Train MAE"], tuned_row["Test MAE"], tuned_row["Train R2"], tuned_row["Test R2"]
    )

    return {
        "Model": model_name,
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
        "Tuned Overfitting Status": tuned_status,
    }
