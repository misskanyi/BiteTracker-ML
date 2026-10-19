# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> **FOOD4EDUCATION RESEARCH RESULTS.** Results below are from the anonymised Food4Education dataset, read live from its Google Sheet. Weather, cost and exam/event fields are not in that sheet and were held constant.

## Why Tuning Was Required

Phase 4 models used library-default hyperparameters, which lets tree-based models fit training noise. Measured default-hyperparameter results: Decision Tree: test R²=0.973 (Overfitting); Random Forest: test R²=0.989 (Overfitting); XGBoost: test R²=0.987 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

## Why TimeSeriesSplit

Meal demand is a time series — a school's future demand must never be predicted using information from its own future. `TimeSeriesSplit` builds growing folds where every validation fold occurs strictly after its training fold, unlike standard k-fold which would shuffle rows and let the model "see" the future during cross-validation.

5 folds were used over the 574 training rows carried over from Phase 4 (the final 20% test set was held out entirely and never passed to GridSearchCV/RandomizedSearchCV). Exact fold ranges:

| Fold | Train Rows | Train Index Range | Val Rows | Val Index Range |
|---|---|---|---|---|
| 1 | 99 | (0, 98) | 95 | (99, 193) |
| 2 | 194 | (0, 193) | 95 | (194, 288) |
| 3 | 289 | (0, 288) | 95 | (289, 383) |
| 4 | 384 | (0, 383) | 95 | (384, 478) |
| 5 | 479 | (0, 478) | 95 | (479, 573) |

## Scoring Strategy

- **Hyperparameters are chosen by cross-validation MAE** (GridSearchCV / RandomizedSearchCV, `refit="mae"`). The operational goal is minimizing the average number of meals mispredicted. Validation RMSE is scored on the same folds.
- **The final model is chosen by lowest TimeSeriesSplit validation MAE** (validation RMSE as tiebreak) across all four candidates. Linear Regression, which has no tuning grid, is cross-validated on the same folds with the same scorers.
- **Test-period metrics are computed only after selection.** The selected model is evaluated once on the untouched test period; test metrics for the other models are reported below for research comparison and did not influence tuning or selection.

## Search Method Per Model

- **Decision Tree** — exhaustive `GridSearchCV` (240 combinations × 5 folds = 1200 fits). Cheap because a single shallow tree fits almost instantly.
- **Random Forest** — `RandomizedSearchCV` (25 of 324 possible combinations). The full grid (up to 300 trees × 324 combinations × 5 folds) was too slow for a development machine; randomized sampling covers the space at a fraction of the cost.
- **XGBoost** — `RandomizedSearchCV` (40 of 4,374 possible combinations, fixed seed 42). The full grid would mean ~21,870 fits — impractical here; randomized search still explores the depth/learning-rate/regularization combinations meant to curb the overfitting seen in Phase 4.
- **Linear Regression** — not tuned (no hyperparameters meaningfully affect ordinary least squares here). It is a candidate ML model, not a naive baseline, and receives a TimeSeriesSplit validation score on the same folds as the others.

## Best Parameters Found

**Decision Tree:**

```json
{
  "max_depth": 8,
  "max_features": null,
  "min_samples_leaf": 10,
  "min_samples_split": 2
}
```

**Random Forest:**

```json
{
  "n_estimators": 300,
  "min_samples_split": 5,
  "min_samples_leaf": 1,
  "max_features": 0.7,
  "max_depth": null,
  "bootstrap": true
}
```

**XGBoost:**

```json
{
  "subsample": 0.85,
  "reg_lambda": 10,
  "reg_alpha": 0.1,
  "n_estimators": 200,
  "min_child_weight": 5,
  "max_depth": 2,
  "learning_rate": 0.05,
  "colsample_bytree": 0.7
}
```

## Tuned Results (Test Set — research comparison, not used for selection)

| Model         |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 | Generalization Gap Status   |   Tuning Time (s) |   Prediction Time (s) |   Best CV RMSE |
|:--------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|:----------------------------|------------------:|----------------------:|---------------:|
| Decision Tree |      51.896 |     61.197 |       88.573 |     111.100 |      0.990 |     0.983 | Moderate Gap                |             4.680 |                 0.001 |        138.921 |
| Random Forest |      27.526 |     49.544 |       47.975 |      88.943 |      0.997 |     0.989 | Overfitting                 |             6.566 |                 0.017 |        131.676 |
| XGBoost       |      50.676 |     51.637 |       84.354 |      88.796 |      0.991 |     0.989 | Good Generalization         |             1.133 |                 0.002 |        124.036 |

## Default-Hyperparameter vs Tuned Comparison (test set, research only)

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              76.846 |           61.197 |            15.649 |              140.834 |           111.100 |             29.734 |              0.973 |           0.983 |            0.010 | Overfitting                   | Moderate Gap               |
| Random Forest |              51.016 |           49.544 |             1.472 |               89.008 |            88.943 |              0.065 |              0.989 |           0.989 |            0.000 | Overfitting                   | Overfitting                |
| XGBoost       |              60.198 |           51.637 |             8.561 |               99.752 |            88.796 |             10.956 |              0.987 |           0.989 |            0.003 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 15.65 meals (Overfitting → Moderate Gap).
- **Random Forest**: Test MAE improved by 1.47 meals (Overfitting → Overfitting).
- **XGBoost**: Test MAE improved by 8.56 meals (Overfitting → Good Generalization).

## Final Model Selection (validation only)

Selection criterion: **lowest TimeSeriesSplit validation MAE**, with lowest TimeSeriesSplit validation RMSE as the tiebreak. Scores are means over the TimeSeriesSplit validation folds of the training period. The test period was not used to rank or select models.

| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |
|---|---|---|---|---|---|
| 1 | XGBoost | 75.916 | 13.489 | 124.036 | 5 |
| 2 | Linear Regression | 78.489 | 13.264 | 124.579 | 5 |
| 3 | Random Forest | 79.228 | 18.572 | 131.676 | 5 |
| 4 | Decision Tree | 82.860 | 20.149 | 138.921 | 5 |

**Selected final model: XGBoost** (validation MAE=75.916).

## Final Test Evaluation (selected model, evaluated once)

- Test MAE: **51.637** meals
- Test RMSE: **88.796** meals
- Test R²: **0.989**
- Train/test generalization: Good Generalization

## Naive Forecasting Baselines

Two simple rules, not fit or tuned, evaluated on the same test period as the ML models. Both use only each school's own realised meal demand from strictly earlier observations:

- **Persistence / Last Observation** — prediction(T) = meals served on the school's most recent available observation before T.
- **7-Day Historical Mean** — prediction(T) = mean meals served over the school's previous 7 available observations (T excluded: shift by one before the rolling window).

Linear Regression is an ML candidate, not a naive baseline. `projected_students` is not used as a baseline.

### Methods Compared

Validation columns are empty for the naive baselines: they were not evaluated on the validation folds. Test columns for non-selected ML models are for research comparison only.

| Method                         | Type           |   Validation MAE |   Validation RMSE |   Selection Rank | Selected   |   Test MAE |   Test RMSE |   Test R2 |
|:-------------------------------|:---------------|-----------------:|------------------:|-----------------:|:-----------|-----------:|------------:|----------:|
| Persistence / Last Observation | Naive baseline |          nan     |           nan     |          nan     | False      |     39.067 |      69.104 |     0.994 |
| 7-Day Historical Mean          | Naive baseline |          nan     |           nan     |          nan     | False      |     58.098 |     119.790 |     0.981 |
| XGBoost                        | ML candidate   |           75.916 |           124.036 |            1.000 | True       |     51.637 |      88.796 |     0.989 |
| Linear Regression              | ML candidate   |           78.489 |           124.579 |            2.000 | False      |     50.914 |      82.666 |     0.991 |
| Random Forest                  | ML candidate   |           79.228 |           131.676 |            3.000 | False      |     49.544 |      88.943 |     0.989 |
| Decision Tree                  | ML candidate   |           82.860 |           138.921 |            4.000 | False      |     61.197 |     111.100 |     0.983 |

### Selected Model vs Naive Baselines (test period)

| Selected Model   | Naive Baseline                 |   Baseline Test MAE |   Model Test MAE |   MAE Improvement |   MAE Improvement (%) |   Baseline Test RMSE |   Model Test RMSE |   RMSE Improvement | Model Beats Baseline (MAE)   |
|:-----------------|:-------------------------------|--------------------:|-----------------:|------------------:|----------------------:|---------------------:|------------------:|-------------------:|:-----------------------------|
| XGBoost          | Persistence / Last Observation |              39.067 |           51.637 |           -12.570 |               -32.176 |               69.104 |            88.796 |            -19.692 | False                        |
| XGBoost          | 7-Day Historical Mean          |              58.098 |           51.637 |             6.461 |                11.121 |              119.790 |            88.796 |             30.994 | True                         |

- vs **Persistence / Last Observation**: XGBoost does NOT beat it: MAE is worse by 12.57 meals (-32.2%).
- vs **7-Day Historical Mean**: XGBoost beats it by 6.46 meals MAE (11.1%).