# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`) and are measured, not fabricated.

## Why Tuning Was Required

Phase 4 baselines used library-default hyperparameters. The Decision Tree overfit severely (train R²=1.0 vs test R²=0.45) and XGBoost also overfit (train R²=0.999 vs test R²=0.768). Random Forest was the best baseline but still had a meaningful train/test gap. This phase tunes depth, leaf-size, regularization, and sampling parameters specifically to reduce that gap, not just to chase a lower training error.

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

- **GridSearchCV / RandomizedSearchCV selection metric: negative MAE.** The operational goal is minimizing the average number of meals mispredicted, so hyperparameters were chosen to minimize cross-validation MAE.
- **RMSE and R² were computed after refitting the best estimator** on the full training set and evaluating once on the untouched test set — they describe the result, they did not influence which hyperparameters were chosen.

## Search Method Per Model

- **Decision Tree** — exhaustive `GridSearchCV` (240 combinations × 5 folds = 1200 fits). Cheap because a single shallow tree fits almost instantly.
- **Random Forest** — `RandomizedSearchCV` (25 of 324 possible combinations). The full grid (up to 300 trees × 324 combinations × 5 folds) was too slow for a development machine; randomized sampling covers the space at a fraction of the cost.
- **XGBoost** — `RandomizedSearchCV` (40 of 4,374 possible combinations, fixed seed 42). The full grid would mean ~21,870 fits — impractical here; randomized search still explores the depth/learning-rate/regularization combinations meant to curb the overfitting seen in Phase 4.
- **Linear Regression** — not tuned, kept as the unchanged Phase 4 baseline reference (no hyperparameters meaningfully affect ordinary least squares here).

## Best Parameters Found

**Decision Tree:**

```json
{
  "max_depth": 5,
  "max_features": "sqrt",
  "min_samples_leaf": 2,
  "min_samples_split": 5
}
```

**Random Forest:**

```json
{
  "n_estimators": 200,
  "min_samples_split": 10,
  "min_samples_leaf": 2,
  "max_features": 0.7,
  "max_depth": null,
  "bootstrap": true
}
```

**XGBoost:**

```json
{
  "subsample": 0.7,
  "reg_lambda": 10,
  "reg_alpha": 0,
  "n_estimators": 200,
  "min_child_weight": 1,
  "max_depth": 2,
  "learning_rate": 0.05,
  "colsample_bytree": 1.0
}
```

## Tuned Results (Test Set)

| Model         |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 | Generalization Gap Status   |   Tuning Time (s) |   Prediction Time (s) |
|:--------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|:----------------------------|------------------:|----------------------:|
| Decision Tree |      30.969 |     32.810 |       69.324 |      77.096 |      0.866 |     0.863 | Good Generalization         |             2.079 |                 0.000 |
| Random Forest |      21.599 |     33.947 |       53.369 |      74.623 |      0.921 |     0.872 | Overfitting                 |             4.477 |                 0.009 |
| XGBoost       |      30.789 |     31.243 |       65.145 |      68.409 |      0.882 |     0.892 | Good Generalization         |             0.886 |                 0.002 |

## Baseline vs Tuned Comparison

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              60.487 |           32.810 |            27.677 |              154.623 |            77.096 |             77.527 |              0.449 |           0.863 |            0.414 | Overfitting                   | Good Generalization        |
| Random Forest |              36.915 |           33.947 |             2.968 |               81.385 |            74.623 |              6.762 |              0.847 |           0.872 |            0.024 | Overfitting                   | Overfitting                |
| XGBoost       |              45.040 |           31.243 |            13.797 |              100.417 |            68.409 |             32.008 |              0.768 |           0.892 |            0.124 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 27.68 meals (Overfitting → Good Generalization).
- **Random Forest**: Test MAE improved by 2.97 meals (Overfitting → Overfitting).
- **XGBoost**: Test MAE improved by 13.80 meals (Overfitting → Good Generalization).

## Final Model Selection

Selection order: (1) lowest test MAE, (2) lowest test RMSE as tiebreak, (3) highest test R² as supporting evidence, (4) acceptable generalization gap, (5) practical prediction speed. The model was **not** chosen for having the highest R² alone.

Ranking by test MAE (best to worst):

1. XGBoost — Test MAE 31.24, Test RMSE 68.41, Test R² 0.892
2. Decision Tree — Test MAE 32.81, Test RMSE 77.10, Test R² 0.863
3. Random Forest — Test MAE 33.95, Test RMSE 74.62, Test R² 0.872
4. Linear Regression — Test MAE 41.43, Test RMSE 82.49, Test R² 0.843

**Selected final model: XGBoost** (Test MAE=31.24, Test RMSE=68.41, Test R²=0.892, generalization: Good Generalization).

## Limitations of Synthetic Data

All tuning and evaluation above used the synthetic dataset generated for development (`src/synthetic_data.py`). Relationships between rainfall, holidays, exams, and attendance were hand-designed to be plausible, not fit to any real county's records. Hyperparameters selected here are only known to be good for this synthetic data's noise structure and scale.

## What to Repeat With Real Data

1. Re-run `run_pipeline.py` and `run_eda.py` against the real dataset once available (point `BITETRACKER_DATA_PATH`/`ACTIVE_DATA_PATH` at it).
2. Re-run `run_training.py` for fresh baselines — real data's noise level and feature relationships will differ from the synthetic generator's assumptions.
3. Re-run this tuning phase (`run_tuning.py`) from scratch — do not reuse the hyperparameters found here; they were fit to synthetic patterns.
4. Re-check the TimeSeriesSplit fold sizes — a real dataset with different row counts or a longer history may warrant more or fewer folds.
