# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> **FOOD4EDUCATION RESEARCH RESULTS.** Results below are from the anonymised Food4Education dataset, read live from its Google Sheet. Weather, cost and exam/event fields are not in that sheet and were held constant.

## Why Tuning Was Required

Phase 4 models used library-default hyperparameters, which lets tree-based models fit training noise. Measured default-hyperparameter results: Decision Tree: test R²=0.986 (Overfitting); Random Forest: test R²=0.990 (Overfitting); XGBoost: test R²=0.988 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

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
  "max_depth": 5,
  "max_features": null,
  "min_samples_leaf": 10,
  "min_samples_split": 2
}
```

**Random Forest:**

```json
{
  "n_estimators": 100,
  "min_samples_split": 5,
  "min_samples_leaf": 1,
  "max_features": 0.7,
  "max_depth": 15,
  "bootstrap": true
}
```

**XGBoost:**

```json
{
  "subsample": 0.85,
  "reg_lambda": 10,
  "reg_alpha": 0,
  "n_estimators": 100,
  "min_child_weight": 5,
  "max_depth": 3,
  "learning_rate": 0.1,
  "colsample_bytree": 0.7
}
```

## Tuned Results (Test Set — research comparison, not used for selection)

| Model         |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 | Generalization Gap Status   |   Tuning Time (s) |   Prediction Time (s) |   Best CV RMSE |
|:--------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|:----------------------------|------------------:|----------------------:|---------------:|
| Decision Tree |      55.457 |     61.233 |       90.859 |     110.196 |      0.990 |     0.984 | Moderate Gap                |             2.238 |                 0.000 |        129.286 |
| Random Forest |      28.297 |     47.778 |       49.923 |      87.003 |      0.997 |     0.990 | Overfitting                 |             2.297 |                 0.003 |        133.327 |
| XGBoost       |      42.779 |     45.068 |       70.485 |      81.139 |      0.994 |     0.991 | Good Generalization         |             0.590 |                 0.002 |        136.219 |

## Default-Hyperparameter vs Tuned Comparison (test set, research only)

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              53.805 |           61.233 |            -7.429 |              104.369 |           110.196 |             -5.827 |              0.986 |           0.984 |           -0.002 | Overfitting                   | Moderate Gap               |
| Random Forest |              46.917 |           47.778 |            -0.861 |               85.429 |            87.003 |             -1.574 |              0.990 |           0.990 |           -0.000 | Overfitting                   | Overfitting                |
| XGBoost       |              56.954 |           45.068 |            11.887 |               96.180 |            81.139 |             15.041 |              0.988 |           0.991 |            0.004 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE did not improve (tuned model performed worse) by -7.43 meals (Overfitting → Moderate Gap).
- **Random Forest**: Test MAE did not improve (tuned model performed worse) by -0.86 meals (Overfitting → Overfitting).
- **XGBoost**: Test MAE improved by 11.89 meals (Overfitting → Good Generalization).

## Final Model Selection (validation only)

Selection criterion: **lowest TimeSeriesSplit validation MAE**, with lowest TimeSeriesSplit validation RMSE as the tiebreak. Scores are means over the TimeSeriesSplit validation folds of the training period. The test period was not used to rank or select models.

| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |
|---|---|---|---|---|---|
| 1 | Random Forest | 78.445 | 14.904 | 133.327 | 5 |
| 2 | Decision Tree | 80.279 | 13.782 | 129.286 | 5 |
| 3 | XGBoost | 80.547 | 12.996 | 136.219 | 5 |
| 4 | Linear Regression | 98.451 | 26.957 | 141.636 | 5 |

**Selected final model: Random Forest** (validation MAE=78.445).

## Final Test Evaluation (selected model, evaluated once)

- Test MAE: **47.778** meals
- Test RMSE: **87.003** meals
- Test R²: **0.990**
- Train/test generalization: Overfitting

## Naive Forecasting Baselines

Two simple rules, not fit or tuned, evaluated on the same test period as the ML models. Both use only each school's own realised meal demand from strictly earlier observations:

- **Persistence / Last Observation** — prediction(T) = meals served on the school's most recent available observation before T.
- **7-Day Historical Mean** — prediction(T) = mean meals served over the school's previous 7 available observations (T excluded: shift by one before the rolling window).

Linear Regression is an ML candidate, not a naive baseline. `projected_students` is not used as a baseline.

### Methods Compared

Validation columns are empty for the naive baselines: they were not evaluated on the validation folds. Test columns for non-selected ML models are for research comparison only.

| Method                         | Type           |   Validation MAE |   Validation RMSE |   Selection Rank | Selected   |   Test MAE |   Test RMSE |   Test R2 |
|:-------------------------------|:---------------|-----------------:|------------------:|-----------------:|:-----------|-----------:|------------:|----------:|
| Persistence / Last Observation | Naive baseline |          nan     |           nan     |          nan     | False      |     40.633 |      74.005 |     0.993 |
| 7-Day Historical Mean          | Naive baseline |          nan     |           nan     |          nan     | False      |     59.989 |     123.876 |     0.980 |
| Random Forest                  | ML candidate   |           78.445 |           133.327 |            1.000 | True       |     47.778 |      87.003 |     0.990 |
| Decision Tree                  | ML candidate   |           80.279 |           129.286 |            2.000 | False      |     61.233 |     110.196 |     0.984 |
| XGBoost                        | ML candidate   |           80.547 |           136.219 |            3.000 | False      |     45.068 |      81.139 |     0.991 |
| Linear Regression              | ML candidate   |           98.451 |           141.636 |            4.000 | False      |     53.242 |      84.620 |     0.991 |

### Selected Model vs Naive Baselines (test period)

| Selected Model   | Naive Baseline                 |   Baseline Test MAE |   Model Test MAE |   MAE Improvement |   MAE Improvement (%) |   Baseline Test RMSE |   Model Test RMSE |   RMSE Improvement | Model Beats Baseline (MAE)   |
|:-----------------|:-------------------------------|--------------------:|-----------------:|------------------:|----------------------:|---------------------:|------------------:|-------------------:|:-----------------------------|
| Random Forest    | Persistence / Last Observation |              40.633 |           47.778 |            -7.146 |               -17.586 |               74.005 |            87.003 |            -12.998 | False                        |
| Random Forest    | 7-Day Historical Mean          |              59.989 |           47.778 |            12.211 |                20.356 |              123.876 |            87.003 |             36.873 | True                         |

- vs **Persistence / Last Observation**: Random Forest does NOT beat it: MAE is worse by 7.15 meals (-17.6%).
- vs **7-Day Historical Mean**: Random Forest beats it by 12.21 meals MAE (20.4%).