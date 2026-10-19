# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> **SYNTHETIC DEVELOPMENT RESULTS.** Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`), not Food4Education or any real school records. They are measured, not fabricated.

## Why Tuning Was Required

Phase 4 models used library-default hyperparameters, which lets tree-based models fit training noise. Measured default-hyperparameter results: Decision Tree: test R²=0.955 (Overfitting); Random Forest: test R²=0.974 (Overfitting); XGBoost: test R²=0.971 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

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
  "max_depth": 8,
  "max_features": null,
  "min_samples_leaf": 1,
  "min_samples_split": 10
}
```

**Random Forest:**

```json
{
  "n_estimators": 100,
  "min_samples_split": 2,
  "min_samples_leaf": 2,
  "max_features": 0.7,
  "max_depth": 10,
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
| Decision Tree |      11.994 |     11.744 |       18.100 |      21.076 |      0.976 |     0.968 | Good Generalization         |             6.633 |                 0.001 |         29.221 |
| Random Forest |      10.307 |     11.377 |       15.046 |      18.974 |      0.983 |     0.974 | Moderate Gap                |            11.796 |                 0.010 |         30.102 |
| XGBoost       |      15.860 |     13.676 |       22.771 |      21.070 |      0.962 |     0.969 | Good Generalization         |             3.607 |                 0.005 |         33.542 |

## Default-Hyperparameter vs Tuned Comparison (test set, research only)

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              14.025 |           11.744 |             2.281 |               25.135 |            21.076 |              4.059 |              0.955 |           0.968 |            0.013 | Overfitting                   | Good Generalization        |
| Random Forest |              11.094 |           11.377 |            -0.283 |               19.131 |            18.974 |              0.157 |              0.974 |           0.974 |            0.000 | Overfitting                   | Moderate Gap               |
| XGBoost       |              12.026 |           13.676 |            -1.650 |               20.200 |            21.070 |             -0.870 |              0.971 |           0.969 |           -0.003 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 2.28 meals (Overfitting → Good Generalization).
- **Random Forest**: Test MAE did not improve (tuned model performed worse) by -0.28 meals (Overfitting → Moderate Gap).
- **XGBoost**: Test MAE did not improve (tuned model performed worse) by -1.65 meals (Overfitting → Good Generalization).

## Final Model Selection (validation only)

Selection criterion: **lowest TimeSeriesSplit validation MAE**, with lowest TimeSeriesSplit validation RMSE as the tiebreak. Scores are means over the TimeSeriesSplit validation folds of the training period. The test period was not used to rank or select models.

| Rank | Model | Validation MAE | Validation MAE Std | Validation RMSE | Folds |
|---|---|---|---|---|---|
| 1 | Decision Tree | 18.716 | 5.961 | 29.221 | 5 |
| 2 | Random Forest | 19.509 | 5.754 | 30.102 | 5 |
| 3 | XGBoost | 23.806 | 9.751 | 33.542 | 5 |
| 4 | Linear Regression | 43.441 | 33.923 | 50.807 | 5 |

**Selected final model: Decision Tree** (validation MAE=18.716).

## Final Test Evaluation (selected model, evaluated once)

- Test MAE: **11.744** meals
- Test RMSE: **21.076** meals
- Test R²: **0.968**
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
| Persistence / Last Observation | Naive baseline |          nan     |           nan     |          nan     | False      |     22.936 |      41.794 |     0.876 |
| 7-Day Historical Mean          | Naive baseline |          nan     |           nan     |          nan     | False      |     29.454 |      51.712 |     0.810 |
| Decision Tree                  | ML candidate   |           18.716 |            29.221 |            1.000 | True       |     11.744 |      21.076 |     0.968 |
| Random Forest                  | ML candidate   |           19.509 |            30.102 |            2.000 | False      |     11.377 |      18.974 |     0.974 |
| XGBoost                        | ML candidate   |           23.806 |            33.542 |            3.000 | False      |     13.676 |      21.070 |     0.969 |
| Linear Regression              | ML candidate   |           43.441 |            50.807 |            4.000 | False      |     18.898 |      26.152 |     0.951 |

### Selected Model vs Naive Baselines (test period)

| Selected Model   | Naive Baseline                 |   Baseline Test MAE |   Model Test MAE |   MAE Improvement |   MAE Improvement (%) |   Baseline Test RMSE |   Model Test RMSE |   RMSE Improvement | Model Beats Baseline (MAE)   |
|:-----------------|:-------------------------------|--------------------:|-----------------:|------------------:|----------------------:|---------------------:|------------------:|-------------------:|:-----------------------------|
| Decision Tree    | Persistence / Last Observation |              22.936 |           11.744 |            11.191 |                48.794 |               41.794 |            21.076 |             20.718 | True                         |
| Decision Tree    | 7-Day Historical Mean          |              29.454 |           11.744 |            17.710 |                60.127 |               51.712 |            21.076 |             30.636 | True                         |

- vs **Persistence / Last Observation**: Decision Tree beats it by 11.19 meals MAE (48.8%).
- vs **7-Day Historical Mean**: Decision Tree beats it by 17.71 meals MAE (60.1%).

## Limitations of Synthetic Data

All tuning and evaluation above used the synthetic dataset generated for development (`src/synthetic_data.py`). Relationships between rainfall, holidays, exams, and attendance were hand-designed to be plausible, not fit to any real county's records. Hyperparameters selected here are only known to be good for this synthetic data's noise structure and scale.

## What to Repeat With Real Data

1. Re-run `run_pipeline.py` and `run_eda.py` against the real dataset once available (point `BITETRACKER_DATA_PATH`/`ACTIVE_DATA_PATH` at it).
2. Re-run `run_training.py` for fresh baselines — real data's noise level and feature relationships will differ from the synthetic generator's assumptions.
3. Re-run this tuning phase (`run_tuning.py`) from scratch — do not reuse the hyperparameters found here; they were fit to synthetic patterns.
4. Re-check the TimeSeriesSplit fold sizes — a real dataset with different row counts or a longer history may warrant more or fewer folds.
