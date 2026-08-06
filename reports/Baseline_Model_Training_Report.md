# BiteTracker — Baseline Model Training Report

> Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`) and are measured, not fabricated — they will change once a real dataset is used.

## Training Process

Four regressors were trained with untuned, library-default hyperparameters (no GridSearchCV — that is Phase 5) to establish a baseline before any tuning effort:

- **Linear Regression** — simple, interpretable baseline; assumes linear relationships between lag/rolling/weather features and next-day demand.
- **Decision Tree Regressor** — captures non-linear splits (e.g. holiday vs term-time thresholds) a linear model can't.
- **Random Forest Regressor** — ensemble of trees, generally more robust and less prone to overfitting than a single tree.
- **XGBoost Regressor** — gradient-boosted trees, often strong on tabular data with mixed categorical/numeric/lag features like this dataset.

## Chronological Split

- Train records: **1200** (2023-01-20 to 2024-08-01)
- Test records: **300** (2024-08-02 to 2024-12-19)
- Split ratio: **80%** train / **20%** test, by row order after date-sorting (no shuffling).
- The test set is strictly the last 20% of chronologically-ordered observations, so every test-set prediction is made using only information from earlier in time.

## Baseline Results (Test Set)

| Model             |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 |   Training Time (s) |
|:------------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|--------------------:|
| Linear Regression |      46.446 |     41.433 |       87.652 |      82.488 |      0.786 |     0.843 |               0.001 |
| Decision Tree     |       0.000 |     60.487 |        0.000 |     154.623 |      1.000 |     0.449 |               0.013 |
| Random Forest     |      13.231 |     36.915 |       29.743 |      81.385 |      0.975 |     0.847 |               0.152 |
| XGBoost           |       2.143 |     45.040 |        3.102 |     100.417 |      1.000 |     0.768 |               0.134 |

- Best by Test MAE: **Random Forest**
- Best by Test RMSE: **Random Forest**
- Best by Test R²: **Random Forest**

## Generalization / Overfitting Check

- **Linear Regression**: Moderate generalization gap (test MAE -11% higher than train MAE, R² gap -0.058) — worth monitoring but not a clear overfitting signal on its own.
- **Decision Tree**: Potential overfitting: test MAE is 6048750000000% higher than train MAE and train R² exceeds test R² by 0.551. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **Random Forest**: Potential overfitting: test MAE is 179% higher than train MAE and train R² exceeds test R² by 0.128. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **XGBoost**: Potential overfitting: test MAE is 2002% higher than train MAE and train R² exceeds test R² by 0.232. The model may be fitting training-period noise that doesn't generalize to the held-out period.

## Why Tuning Is Still Required

These are default-hyperparameter baselines. Tree-based models in particular (Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` to search a small, practical hyperparameter grid per model without shuffling or leaking future data into cross-validation folds. Whichever model looks best here may or may not remain best after tuning.

## Synthetic Data Caveat

All figures in this report come from the synthetic dataset generated for development (see `src/synthetic_data.py`), not real school records. Relationships (e.g. rainfall's effect on attendance) were designed to be plausible but are not calibrated to any real county's data. Treat these results as a pipeline validation exercise, not a claim about real-world model performance.
