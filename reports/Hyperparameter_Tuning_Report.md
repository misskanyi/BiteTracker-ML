# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> **SYNTHETIC DEVELOPMENT RESULTS.** Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`), not Food4Education or any real school records. They are measured, not fabricated.

## Why Tuning Was Required

Phase 4 models used library-default hyperparameters, which lets tree-based models fit training noise. Measured default-hyperparameter results: Decision Tree: test R²=0.894 (Overfitting); Random Forest: test R²=0.961 (Overfitting); XGBoost: test R²=0.901 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

## Why TimeSeriesSplit

Meal demand is a time series — a school's future demand must never be predicted using information from its own future. `TimeSeriesSplit` builds growing folds where every validation fold occurs strictly after its training fold, unlike standard k-fold which would shuffle rows and let the model "see" the future during cross-validation.

5 folds were used over the 1200 training rows carried over from Phase 4 (the final 20% test set was held out entirely and never passed to GridSearchCV/RandomizedSearchCV). Exact fold ranges:

| Fold | Train Rows | Train Index Range | Val Rows | Val Index Range |
|---|---|---|---|---|
| 1 | 200 | (0, 199) | 200 | (200, 399) |
| 2 | 400 | (0, 399) | 200 | (400, 599) |
| 3 | 600 | (0, 599) | 200 | (600, 799) |
| 4 | 800 | (0, 799) | 200 | (800, 999) |
| 5 | 1000 | (0, 999) | 200 | (1000, 1199) |

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
  "min_samples_split": 2,
  "min_samples_leaf": 4,
  "max_features": "sqrt",
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
  "n_estimators": 200,
  "min_child_weight": 3,
  "max_depth": 2,
  "learning_rate": 0.05,
  "colsample_bytree": 0.85
}
```

## Tuned Results (Test Set — research comparison, not used for selection)

| Model         |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 | Generalization Gap Status   |   Tuning Time (s) |   Prediction Time (s) |   Best CV RMSE |
|:--------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|:----------------------------|------------------:|----------------------:|---------------:|
| Decision Tree |      17.650 |     15.103 |       36.429 |      39.268 |      0.963 |     0.965 | Good Generalization         |             5.158 |                 0.001 |        113.662 |
| Random Forest |      16.243 |     19.370 |       38.510 |      47.072 |      0.958 |     0.949 | Moderate Gap                |             8.249 |                 0.007 |         82.405 |
| XGBoost       |      17.542 |     15.656 |       35.600 |      39.675 |      0.964 |     0.964 | Good Generalization         |             2.050 |                 0.003 |         92.294 |

## Default-Hyperparameter vs Tuned Comparison (test set, research only)

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              21.644 |           15.103 |             6.541 |               67.971 |            39.268 |             28.703 |              0.894 |           0.965 |            0.071 | Overfitting                   | Good Generalization        |
| Random Forest |              15.336 |           19.370 |            -4.034 |               41.315 |            47.072 |             -5.757 |              0.961 |           0.949 |           -0.012 | Overfitting                   | Moderate Gap               |
| XGBoost       |              19.196 |           15.656 |             3.541 |               65.763 |            39.675 |             26.088 |              0.901 |           0.964 |            0.063 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 6.54 meals (Overfitting → Good Generalization).
- **Random Forest**: Test MAE did not improve (tuned model performed worse) by -4.03 meals (Overfitting → Moderate Gap).
- **XGBoost**: Test MAE improved by 3.54 meals (Overfitting → Good Generalization).

## Final Model Selection (validation only)

Selection criterion: **lowest TimeSeriesSplit validation MAE**, with lowest TimeSeriesSplit validation RMSE as the tiebreak. Scores are means over the TimeSeriesSplit validation folds of the training period. The test period was not used to rank or select models.

| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |
|---|---|---|---|---|---|
| 1 | Random Forest | 55.430 | 48.755 | 82.405 | 5 |
| 2 | Linear Regression | 60.543 | 79.932 | 79.568 | 5 |
| 3 | XGBoost | 71.313 | 102.225 | 92.294 | 5 |
| 4 | Decision Tree | 90.895 | 144.190 | 113.662 | 5 |

**Selected final model: Random Forest** (validation MAE=55.430).

## Final Test Evaluation (selected model, evaluated once)

- Test MAE: **19.370** meals
- Test RMSE: **47.072** meals
- Test R²: **0.949**
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
| Persistence / Last Observation | Naive baseline |          nan     |           nan     |          nan     | False      |     31.434 |      81.188 |     0.849 |
| 7-Day Historical Mean          | Naive baseline |          nan     |           nan     |          nan     | False      |     58.906 |     122.915 |     0.653 |
| Random Forest                  | ML candidate   |           55.430 |            82.405 |            1.000 | True       |     19.370 |      47.072 |     0.949 |
| Linear Regression              | ML candidate   |           60.543 |            79.568 |            2.000 | False      |     17.171 |      38.881 |     0.965 |
| XGBoost                        | ML candidate   |           71.313 |            92.294 |            3.000 | False      |     15.656 |      39.675 |     0.964 |
| Decision Tree                  | ML candidate   |           90.895 |           113.662 |            4.000 | False      |     15.103 |      39.268 |     0.965 |

### Selected Model vs Naive Baselines (test period)

| Selected Model   | Naive Baseline                 |   Baseline Test MAE |   Model Test MAE |   MAE Improvement |   MAE Improvement (%) |   Baseline Test RMSE |   Model Test RMSE |   RMSE Improvement | Model Beats Baseline (MAE)   |
|:-----------------|:-------------------------------|--------------------:|-----------------:|------------------:|----------------------:|---------------------:|------------------:|-------------------:|:-----------------------------|
| Random Forest    | Persistence / Last Observation |              31.434 |           19.370 |            12.064 |                38.380 |               81.188 |            47.072 |             34.116 | True                         |
| Random Forest    | 7-Day Historical Mean          |              58.906 |           19.370 |            39.536 |                67.118 |              122.915 |            47.072 |             75.844 | True                         |

- vs **Persistence / Last Observation**: Random Forest beats it by 12.06 meals MAE (38.4%).
- vs **7-Day Historical Mean**: Random Forest beats it by 39.54 meals MAE (67.1%).

## Limitations of Synthetic Data

All tuning and evaluation above used the synthetic dataset generated for development (`src/synthetic_data.py`). Relationships between rainfall, holidays, exams, and attendance were hand-designed to be plausible, not fit to any real county's records. Hyperparameters selected here are only known to be good for this synthetic data's noise structure and scale.

## What to Repeat With Real Data

1. Re-run `run_pipeline.py` and `run_eda.py` against the real dataset once available (point `BITETRACKER_DATA_PATH`/`ACTIVE_DATA_PATH` at it).
2. Re-run `run_training.py` for fresh baselines — real data's noise level and feature relationships will differ from the synthetic generator's assumptions.
3. Re-run this tuning phase (`run_tuning.py`) from scratch — do not reuse the hyperparameters found here; they were fit to synthetic patterns.
4. Re-check the TimeSeriesSplit fold sizes — a real dataset with different row counts or a longer history may warrant more or fewer folds.
