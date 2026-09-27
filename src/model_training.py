"""Baseline model training for next-school-day meal-demand forecasting.

Trains four regressors with sensible defaults (no hyperparameter tuning —
that's Phase 5) on a chronological 80/20 split, so their untuned behaviour
establishes a baseline before any GridSearchCV work.
"""

# In simple terms: teaches 4 different models (Linear Regression, Decision
# Tree, Random Forest, XGBoost) to guess tomorrow's meal count, using
# default settings, so we have a starting point to compare against later.

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config import settings
from src.feature_engineering import FEATURE_COLUMNS
from src.utils import get_logger

logger = get_logger(__name__)

MODEL_FILENAMES = {
    "linear_regression": "linear_regression.joblib",
    "decision_tree": "decision_tree.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model feature matrix X and target vector y from a processed DataFrame."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing expected feature columns: {missing}")
    X = df[FEATURE_COLUMNS].copy()
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)
    y = df[settings.TARGET_COLUMN].copy()
    return X, y


def chronological_split(
    df: pd.DataFrame, train_ratio: float = settings.TRAIN_TEST_SPLIT_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-sorted DataFrame by row order (not date value) into 80/20.

    Unlike `model_evaluation.chronological_train_test_split` (which splits on
    the date range so every school shares the same cutoff date), this splits
    strictly on row position after sorting by date, guaranteeing the last
    20% of *rows* form the test set regardless of how many schools report on
    any given day. Both approaches avoid shuffling and future leakage; this
    one is used here because it's what "last 20% of observations" means row-wise.
    """
    df = df.sort_values(settings.DATE_COLUMN).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit ordinary least squares.

    Lag/rolling meal-count features are correlated with each other, which
    makes the design matrix moderately ill-conditioned; NumPy's internal
    lstsq solver emits benign RuntimeWarnings from an intermediate step even
    though the final coefficients and predictions stay finite. Suppressed
    here rather than left to clutter output — see module docstring.
    """
    import warnings

    model = LinearRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(random_state=settings.RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=settings.RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(random_state=settings.RANDOM_SEED, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    return model


TRAINERS = {
    "linear_regression": train_linear_regression,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def save_model(model, name: str, models_dir: Path | None = None) -> Path:
    """Persist a trained model to models/ using its canonical filename."""
    out_dir = models_dir or settings.MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MODEL_FILENAMES[name]
    joblib.dump(model, path)
    logger.info("Saved %s to %s", name, path)
    return path


def load_model(name: str, models_dir: Path | None = None):
    """Load a previously saved model by its short name (e.g. 'random_forest')."""
    models_dir = models_dir or settings.MODELS_DIR
    path = models_dir / MODEL_FILENAMES[name]
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    return joblib.load(path)
