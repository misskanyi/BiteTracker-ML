# BiteTracker — Baseline Model Training Report

> **SYNTHETIC DEVELOPMENT RESULTS.** Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`), not Food4Education records, and are measured, not fabricated — they will change once a real dataset is used.

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
| Linear Regression |      68.369 |     53.242 |      109.380 |      84.620 |      0.985 |     0.991 |               0.011 |
| Decision Tree     |       0.000 |     53.805 |        0.000 |     104.369 |      1.000 |     0.986 |               0.003 |
| Random Forest     |      23.945 |     46.917 |       41.656 |      85.429 |      0.998 |     0.990 |               0.063 |
| XGBoost           |       1.118 |     56.954 |        1.586 |      96.180 |      1.000 |     0.988 |               0.117 |

- Best by Test MAE: **Random Forest**
- Best by Test RMSE: **Linear Regression**
- Best by Test R²: **Linear Regression**

## Generalization / Overfitting Check

- **Linear Regression**: Train/test performance are close — no strong evidence of overfitting.
- **Decision Tree**: Potential overfitting: test MAE is 5380489864865% higher than train MAE and train R² exceeds test R² by 0.014. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **Random Forest**: Potential overfitting: test MAE is 96% higher than train MAE and train R² exceeds test R² by 0.008. The model may be fitting training-period noise that doesn't generalize to the held-out period.
- **XGBoost**: Potential overfitting: test MAE is 4993% higher than train MAE and train R² exceeds test R² by 0.012. The model may be fitting training-period noise that doesn't generalize to the held-out period.

## Why Tuning Is Still Required

These are default-hyperparameter baselines. Tree-based models in particular (Decision Tree, Random Forest, XGBoost) are sensitive to depth, leaf-size, and learning-rate settings — Phase 5 applies `GridSearchCV` with `TimeSeriesSplit` to search a small, practical hyperparameter grid per model without shuffling or leaking future data into cross-validation folds. Whichever model looks best here may or may not remain best after tuning.

## Synthetic Data Caveat

All figures in this report come from the synthetic dataset generated for development (see `src/synthetic_data.py`), not real school records. Relationships (e.g. rainfall's effect on attendance) were designed to be plausible but are not calibrated to any real county's data. Treat these results as a pipeline validation exercise, not a claim about real-world model performance.
