# BiteTracker — Baseline Model Training Report

> **FOOD4EDUCATION RESEARCH RESULTS.** Results below are from the anonymised Food4Education dataset, read live from its Google Sheet. Weather, cost and exam/event fields are not in that sheet and were held constant.

## Training Process

Four ML regressors were trained with untuned, library-default hyperparameters (no GridSearchCV — that is Phase 5) as a reference point before any tuning effort. These are default-hyperparameter ML models, not naive forecasting baselines: the naive baselines (Persistence and 7-Day Historical Mean) are reported in `Hyperparameter_Tuning_Report.md`. Test-set results here are descriptive only; the final model is selected in Phase 5 by TimeSeriesSplit validation MAE.

- **Linear Regression** — simple, interpretable ML candidate; assumes linear relationships between lag/rolling/weather features and a school day's demand.
- **Decision Tree Regressor** — captures non-linear splits (e.g. holiday vs term-time thresholds) a linear model can't.
- **Random Forest Regressor** — ensemble of trees, generally more robust and less prone to overfitting than a single tree.
- **XGBoost Regressor** — gradient-boosted trees, often strong on tabular data with mixed categorical/numeric/lag features like this dataset.

## Chronological Split

- Train records: **1982** (2023-02-10 to 2025-04-29)
- Test records: **500** (2025-04-30 to 2025-11-03)
- Split ratio: **80%** train / **20%** test, by row order after date-sorting (no shuffling).
- The test set is strictly the last 20% of chronologically-ordered observations, so every test-set prediction is made using only information from earlier in time.

## Baseline Results (Test Set)

| Model             |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 |   Training Time (s) |
|:------------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|--------------------:|
| Linear Regression |     105.280 |     80.106 |      170.676 |     135.131 |      0.959 |     0.976 |               0.012 |
| Decision Tree     |       0.000 |    126.308 |        0.000 |     213.829 |      1.000 |     0.940 |               0.019 |
| Random Forest     |      36.597 |     80.888 |       61.452 |     137.355 |      0.995 |     0.975 |               0.280 |
| XGBoost           |      12.879 |    103.758 |       18.437 |     167.874 |      1.000 |     0.963 |               0.376 |

- Best by Test MAE: **Linear Regression**
- Best by Test RMSE: **Linear Regression**
- Best by Test R²: **Linear Regression**

## Generalization / Overfitting Check

- **Linear Regression**: Train/test performance are close — no strong evidence of overfitting.
- **Decision Tree**: Potential overfitting: test MAE is 126.3 meals higher than train MAE (126.3 vs 0.0) and train R² exceeds test R² by 0.060. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **Random Forest**: Potential overfitting: test MAE is 44.3 meals higher than train MAE (80.9 vs 36.6) and train R² exceeds test R² by 0.019. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **XGBoost**: Potential overfitting: test MAE is 90.9 meals higher than train MAE (103.8 vs 12.9) and train R² exceeds test R² by 0.037. The model may be fitting training-period noise that doesn't generalize to the held-out period.

## Why Tuning Is Still Required

These are default-hyperparameter baselines. Tree-based models in particular (Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` to search a small, practical hyperparameter grid per model without shuffling or leaking future data into cross-validation folds. Whichever model looks best here may or may not remain best after tuning.
