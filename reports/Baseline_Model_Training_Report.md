# BiteTracker — Baseline Model Training Report

> Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`) and are measured, not fabricated — they will change once a real dataset is used.

## Training Process

Four regressors were trained with untuned, library-default hyperparameters (no GridSearchCV — that is Phase 5) to establish a baseline before any tuning effort:

- **Linear Regression** — simple, interpretable baseline; assumes linear relationships between lag/rolling/weather features and a school day's demand.
- **Decision Tree Regressor** — captures non-linear splits (e.g. holiday vs term-time thresholds) a linear model can't.
- **Random Forest Regressor** — ensemble of trees, generally more robust and less prone to overfitting than a single tree.
- **XGBoost Regressor** — gradient-boosted trees, often strong on tabular data with mixed categorical/numeric/lag features like this dataset.

## Chronological Split

- Train records: **1200** (2023-01-20 to 2024-08-01)
- Test records: **303** (2024-08-02 to 2024-12-20)
- Split ratio: **80%** train / **20%** test, by row order after date-sorting (no shuffling).
- The test set is strictly the last 20% of chronologically-ordered observations, so every test-set prediction is made using only information from earlier in time.

## Baseline Results (Test Set)

| Model             |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 |   Training Time (s) |
|:------------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|--------------------:|
| Linear Regression |      19.317 |     17.171 |       39.428 |      38.881 |      0.956 |     0.965 |               0.002 |
| Decision Tree     |       0.000 |     21.644 |        0.000 |      67.971 |      1.000 |     0.894 |               0.016 |
| Random Forest     |       6.626 |     15.336 |       15.155 |      41.315 |      0.994 |     0.961 |               0.164 |
| XGBoost           |       1.257 |     19.196 |        1.776 |      65.763 |      1.000 |     0.901 |               0.181 |

- Best by Test MAE: **Random Forest**
- Best by Test RMSE: **Linear Regression**
- Best by Test R²: **Linear Regression**

## Generalization / Overfitting Check

- **Linear Regression**: Train/test performance are close — no strong evidence of overfitting.
- **Decision Tree**: Potential overfitting: test MAE is 2164356435644% higher than train MAE and train R² exceeds test R² by 0.106. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **Random Forest**: Potential overfitting: test MAE is 131% higher than train MAE and train R² exceeds test R² by 0.033. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **XGBoost**: Potential overfitting: test MAE is 1427% higher than train MAE and train R² exceeds test R² by 0.099. The model may be fitting training-period noise that doesn't generalize to the held-out period.

## Why Tuning Is Still Required

These are default-hyperparameter baselines. Tree-based models in particular (Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` to search a small, practical hyperparameter grid per model without shuffling or leaking future data into cross-validation folds. Whichever model looks best here may or may not remain best after tuning.

## Synthetic Data Caveat

All figures in this report come from the synthetic dataset generated for development (see `src/synthetic_data.py`), not real school records. Relationships (e.g. rainfall's effect on attendance) were designed to be plausible but are not calibrated to any real county's data. Treat these results as a pipeline validation exercise, not a claim about real-world model performance.
