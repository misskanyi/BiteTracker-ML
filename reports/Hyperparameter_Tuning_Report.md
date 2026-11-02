# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> **FOOD4EDUCATION RESEARCH RESULTS.** Results below are from the anonymised Food4Education dataset, read live from its Google Sheet. Weather, cost and exam/event fields are not in that sheet and were held constant.

## Why Tuning Was Required

Phase 4 models used library-default hyperparameters, which lets tree-based models fit training noise. Measured default-hyperparameter results: Decision Tree: test R²=0.940 (Overfitting); Random Forest: test R²=0.975 (Overfitting); XGBoost: test R²=0.963 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

## Why TimeSeriesSplit

Meal demand is a time series — a school's future demand must never be predicted using information from its own future. `TimeSeriesSplit` builds growing folds where every validation fold occurs strictly after its training fold, unlike standard k-fold which would shuffle rows and let the model "see" the future during cross-validation.

5 folds were used over the 1982 training rows carried over from Phase 4 (the final 20% test set was held out entirely and never passed to GridSearchCV/RandomizedSearchCV). Exact fold ranges:

| Fold | Train Rows | Train Index Range | Val Rows | Val Index Range |
|---|---|---|---|---|
| 1 | 332 | (0, 331) | 330 | (332, 661) |
| 2 | 662 | (0, 661) | 330 | (662, 991) |
| 3 | 992 | (0, 991) | 330 | (992, 1321) |
| 4 | 1322 | (0, 1321) | 330 | (1322, 1651) |
| 5 | 1652 | (0, 1651) | 330 | (1652, 1981) |

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
  "n_estimators": 300,
  "min_samples_split": 10,
  "min_samples_leaf": 1,
  "max_features": "log2",
  "max_depth": 10,
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
| Decision Tree |      92.661 |     93.488 |      153.186 |     162.368 |      0.967 |     0.965 | Good Generalization         |             7.041 |                 0.001 |        182.235 |
| Random Forest |      74.792 |     83.320 |      122.744 |     136.176 |      0.979 |     0.976 | Moderate Gap                |            13.574 |                 0.023 |        173.604 |
| XGBoost       |      84.581 |     83.763 |      138.581 |     134.761 |      0.973 |     0.976 | Good Generalization         |             1.604 |                 0.003 |        177.334 |

## Default-Hyperparameter vs Tuned Comparison (test set, research only)

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |             126.308 |           93.488 |            32.820 |              213.829 |           162.368 |             51.461 |              0.940 |           0.965 |            0.026 | Overfitting                   | Good Generalization        |
| Random Forest |              80.888 |           83.320 |            -2.433 |              137.355 |           136.176 |              1.179 |              0.975 |           0.976 |            0.000 | Overfitting                   | Moderate Gap               |
| XGBoost       |             103.758 |           83.763 |            19.995 |              167.874 |           134.761 |             33.112 |              0.963 |           0.976 |            0.013 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 32.82 meals (Overfitting → Good Generalization).
- **Random Forest**: Test MAE did not improve (tuned model performed worse) by -2.43 meals (Overfitting → Moderate Gap).
- **XGBoost**: Test MAE improved by 20.00 meals (Overfitting → Good Generalization).

## Final Model Selection (validation only)

Selection criterion: **lowest TimeSeriesSplit validation MAE**, with lowest TimeSeriesSplit validation RMSE as the tiebreak. Scores are means over the TimeSeriesSplit validation folds of the training period. The test period was not used to rank or select models.

| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |
|---|---|---|---|---|---|
| 1 | Random Forest | 105.302 | 9.773 | 173.604 | 5 |
| 2 | XGBoost | 106.620 | 6.744 | 177.334 | 5 |
| 3 | Linear Regression | 109.431 | 9.413 | 178.678 | 5 |
| 4 | Decision Tree | 111.514 | 7.047 | 182.235 | 5 |

**Selected final model: Random Forest** (validation MAE=105.302).

## Final Test Evaluation (selected model, evaluated once)

- Test MAE: **83.320** meals
- Test RMSE: **136.176** meals
- Test R²: **0.976**
- Train/test generalization: Moderate Gap

## Naive Forecasting Baselines

Two simple rules, not fit or tuned, evaluated on the same test period as the ML models. Both use only each school's own realised meal demand from strictly earlier observations:

- **Persistence / Last Observation** — prediction(T) = meals served on the school's most recent available observation before T.
- **7-Day Historical Mean** — prediction(T) = mean meals served over the school's previous 7 available observations (T excluded: shift by one before the rolling window).

Linear Regression is an ML candidate, not a naive baseline. `projected_students` is not used as a baseline.

### Methods Compared

Validation columns are empty for the naive baselines: they were not evaluated on the validation folds. Test columns for non-selected ML models are for research comparison only.

| Method                         | Type           |   Validation MAE |   Validation RMSE |   Selection Rank | Selected   |   Test MAE |   Test RMSE |   Test R2 |
|:-------------------------------|:---------------|-----------------:|------------------:|-----------------:|:-----------|-----------:|------------:|----------:|
| Persistence / Last Observation | Naive baseline |          nan     |           nan     |          nan     | False      |     65.500 |     126.811 |     0.979 |
| 7-Day Historical Mean          | Naive baseline |          nan     |           nan     |          nan     | False      |     83.318 |     162.125 |     0.965 |
| Random Forest                  | ML candidate   |          105.302 |           173.604 |            1.000 | True       |     83.320 |     136.176 |     0.976 |
| XGBoost                        | ML candidate   |          106.620 |           177.334 |            2.000 | False      |     83.763 |     134.761 |     0.976 |
| Linear Regression              | ML candidate   |          109.431 |           178.678 |            3.000 | False      |     80.106 |     135.131 |     0.976 |
| Decision Tree                  | ML candidate   |          111.514 |           182.235 |            4.000 | False      |     93.488 |     162.368 |     0.965 |

### Selected Model vs Naive Baselines (test period)

| Selected Model   | Naive Baseline                 |   Baseline Test MAE |   Model Test MAE |   MAE Improvement |   MAE Improvement (%) |   Baseline Test RMSE |   Model Test RMSE |   RMSE Improvement | Model Beats Baseline (MAE)   |
|:-----------------|:-------------------------------|--------------------:|-----------------:|------------------:|----------------------:|---------------------:|------------------:|-------------------:|:-----------------------------|
| Random Forest    | Persistence / Last Observation |              65.500 |           83.320 |           -17.821 |               -27.207 |              126.811 |           136.176 |             -9.365 | False                        |
| Random Forest    | 7-Day Historical Mean          |              83.318 |           83.320 |            -0.003 |                -0.003 |              162.125 |           136.176 |             25.949 | False                        |

- vs **Persistence / Last Observation**: Random Forest does NOT beat it: MAE is worse by 17.82 meals (-27.2%).
- vs **7-Day Historical Mean**: Random Forest does NOT beat it: MAE is worse by 0.00 meals (-0.0%).