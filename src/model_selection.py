"""Final model selection from TimeSeriesSplit validation scores only.

The held-out test period must not influence which model wins, so the
selector here only accepts `ValidationScore` records — a type that has no
field for test metrics. Test-set evaluation happens afterwards, on the one
selected model (see run_tuning.evaluate_selected_model).

Every candidate is scored on the same TimeSeriesSplit folds of the training
period:
- tuned models (Decision Tree, Random Forest, XGBoost): the cross-validation
  scores of the best hyperparameter combination found by the search;
- Linear Regression (no tuning grid): one `cross_validate` run on the same
  folds with the same scorers.

Ranking: lowest mean validation MAE, validation RMSE as tiebreak.
"""

# In simple terms: picks the winning model using only its scores on
# practice (validation) periods inside the training data — never the
# final exam (test set).

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate

from src.utils import get_logger

logger = get_logger(__name__)

SELECTION_CRITERION = "lowest TimeSeriesSplit validation MAE"
TIEBREAK_CRITERION = "lowest TimeSeriesSplit validation RMSE"

# Scorers shared by the hyperparameter searches and the Linear Regression CV,
# so every candidate's validation score is computed the same way.
CV_SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
CV_REFIT_METRIC = "mae"


@dataclass(frozen=True)
class ValidationScore:
    """A candidate model's cross-validation score on the training period.

    Deliberately has no test-metric fields: this is the only input the
    final-model selector accepts.
    """

    model: str
    cv_mae: float
    cv_rmse: float | None
    cv_mae_std: float
    n_folds: int


def validation_score_from_search(model: str, search) -> ValidationScore:
    """Read the best candidate's CV scores from a fitted GridSearchCV/RandomizedSearchCV.

    Works with a single-metric search (scoring="neg_mean_absolute_error")
    or the multi-metric CV_SCORING search; RMSE is None when not scored.
    """
    results = search.cv_results_
    idx = search.best_index_
    if "mean_test_mae" in results:
        mae_key, std_key = "mean_test_mae", "std_test_mae"
    else:
        mae_key, std_key = "mean_test_score", "std_test_score"
    cv_rmse = -float(results["mean_test_rmse"][idx]) if "mean_test_rmse" in results else None
    return ValidationScore(
        model=model,
        cv_mae=-float(results[mae_key][idx]),
        cv_rmse=cv_rmse,
        cv_mae_std=float(results[std_key][idx]),
        n_folds=int(search.n_splits_),
    )


def cross_validate_linear_regression(
    X_train: pd.DataFrame, y_train: pd.Series, cv: TimeSeriesSplit
) -> ValidationScore:
    """Score Linear Regression on the same TimeSeriesSplit folds as the tuned models.

    Linear Regression has no hyperparameter grid, so this is a single
    configuration evaluated fold by fold, with the same scorers the searches use.
    The RuntimeWarning suppression matches model_training.train_linear_regression.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = cross_validate(LinearRegression(), X_train, y_train, cv=cv, scoring=CV_SCORING)
    mae = -scores["test_mae"]
    rmse = -scores["test_rmse"]
    return ValidationScore(
        model="Linear Regression",
        cv_mae=float(np.mean(mae)),
        cv_rmse=float(np.mean(rmse)),
        cv_mae_std=float(np.std(mae)),
        n_folds=len(mae),
    )


def rank_by_validation(scores: list[ValidationScore]) -> list[ValidationScore]:
    """Order candidates by validation MAE, then validation RMSE (missing RMSE sorts last)."""
    if not scores:
        raise ValueError("No candidate validation scores to rank")
    for score in scores:
        if not isinstance(score, ValidationScore):
            raise TypeError(
                "Model selection only accepts ValidationScore records (training-period CV scores); "
                f"got {type(score).__name__}. Test-set metrics must not be used for selection."
            )
        if not math.isfinite(score.cv_mae):
            raise ValueError(f"Validation MAE for {score.model} is not finite: {score.cv_mae}")
    return sorted(scores, key=lambda s: (s.cv_mae, s.cv_rmse if s.cv_rmse is not None else math.inf))


def select_model_by_validation(scores: list[ValidationScore]) -> tuple[ValidationScore, list[ValidationScore]]:
    """Pick the final model: lowest TimeSeriesSplit validation MAE, validation RMSE as tiebreak.

    Returns (selected, ranked). Takes no test data and no test metrics.
    """
    ranked = rank_by_validation(scores)
    selected = ranked[0]
    logger.info("Validation ranking (MAE): %s", [f"{s.model}={s.cv_mae:.3f}" for s in ranked])
    logger.info("Selected %s by %s (%.3f)", selected.model, SELECTION_CRITERION, selected.cv_mae)
    return selected, ranked
