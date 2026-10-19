# BiteTracker — Baseline Model Training Report

> **FOOD4EDUCATION RESEARCH RESULTS.** Results below are from the anonymised Food4Education dataset, read live from its Google Sheet. Weather, cost and exam/event fields are not in that sheet and were held constant.

## Training Process

Four ML regressors were trained with untuned, library-default hyperparameters (no GridSearchCV — that is Phase 5) as a reference point before any tuning effort. These are default-hyperparameter ML models, not naive forecasting baselines: the naive baselines (Persistence and 7-Day Historical Mean) are reported in `Hyperparameter_Tuning_Report.md`. Test-set results here are descriptive only; the final model is selected in Phase 5 by TimeSeriesSplit validation MAE.

- **Linear Regression** — simple, interpretable ML candidate; assumes linear relationships between lag/rolling/weather features and a school day's demand.
- **Decision Tree Regressor** — captures non-linear splits (e.g. holiday vs term-time thresholds) a linear model can't.
- **Random Forest Regressor** — ensemble of trees, generally more robust and less prone to overfitting than a single tree.
- **XGBoost Regressor** — gradient-boosted trees, often strong on tabular data with mixed categorical/numeric/lag features like this dataset.

## Chronological Split

- Train records: **574** (2025-01-24 to 2025-09-09)
- Test records: **148** (2025-09-10 to 2025-11-03)
- Split ratio: **80%** train / **20%** test, by row order after date-sorting (no shuffling).
- The test set is strictly the last 20% of chronologically-ordered observations, so every test-set prediction is made using only information from earlier in time.

## Baseline Results (Test Set)

| Model             |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 |   Training Time (s) |
|:------------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|--------------------:|
| Linear Regression |      65.507 |     50.914 |      107.156 |      82.666 |      0.986 |     0.991 |               0.001 |
| Decision Tree     |       0.000 |     76.846 |        0.000 |     140.834 |      1.000 |     0.973 |               0.004 |
| Random Forest     |      23.036 |     51.016 |       39.106 |      89.008 |      0.998 |     0.989 |               0.089 |
| XGBoost           |       1.184 |     60.198 |        1.706 |      99.752 |      1.000 |     0.987 |               0.164 |

- Best by Test MAE: **Linear Regression**
- Best by Test RMSE: **Linear Regression**
- Best by Test R²: **Linear Regression**

## Generalization / Overfitting Check

- **Linear Regression**: Train/test performance are close — no strong evidence of overfitting.
- **Decision Tree**: Potential overfitting: test MAE is 76.8 meals higher than train MAE (76.8 vs 0.0) and train R² exceeds test R² by 0.027. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **Random Forest**: Potential overfitting: test MAE is 28.0 meals higher than train MAE (51.0 vs 23.0) and train R² exceeds test R² by 0.009. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **XGBoost**: Potential overfitting: test MAE is 59.0 meals higher than train MAE (60.2 vs 1.2) and train R² exceeds test R² by 0.013. The model may be fitting training-period noise that doesn't generalize to the held-out period.

## Why Tuning Is Still Required

These are default-hyperparameter baselines. Tree-based models in particular (Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` to search a small, practical hyperparameter grid per model without shuffling or leaking future data into cross-validation folds. Whichever model looks best here may or may not remain best after tuning.
