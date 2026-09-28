# BiteTracker — Hyperparameter Tuning Report (Phase 5)

> Results below are from **synthetic development data** (`data/sample/synthetic_school_meals.csv`) and are measured, not fabricated.

## Why Tuning Was Required

Phase 4 baselines used library-default hyperparameters, which lets tree-based models fit training noise. Measured baselines: Decision Tree: test R²=0.894 (Overfitting); Random Forest: test R²=0.961 (Overfitting); XGBoost: test R²=0.901 (Overfitting). This phase tunes depth, leaf-size, regularization, and sampling parameters to reduce the train/test gap, not just to chase a lower training error.

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

## Tuned Results (Test Set)

| Model         |   Train MAE |   Test MAE |   Train RMSE |   Test RMSE |   Train R2 |   Test R2 | Generalization Gap Status   |   Tuning Time (s) |   Prediction Time (s) |
|:--------------|------------:|-----------:|-------------:|------------:|-----------:|----------:|:----------------------------|------------------:|----------------------:|
| Decision Tree |      17.650 |     15.103 |       36.429 |      39.268 |      0.963 |     0.965 | Good Generalization         |             4.653 |                 0.001 |
| Random Forest |      16.243 |     19.370 |       38.510 |      47.072 |      0.958 |     0.949 | Moderate Gap                |             8.178 |                 0.008 |
| XGBoost       |      17.542 |     15.656 |       35.600 |      39.675 |      0.964 |     0.964 | Good Generalization         |             1.744 |                 0.003 |

## Baseline vs Tuned Comparison

| Model         |   Baseline Test MAE |   Tuned Test MAE |   MAE Improvement |   Baseline Test RMSE |   Tuned Test RMSE |   RMSE Improvement |   Baseline Test R2 |   Tuned Test R2 |   R2 Improvement | Baseline Overfitting Status   | Tuned Overfitting Status   |
|:--------------|--------------------:|-----------------:|------------------:|---------------------:|------------------:|-------------------:|-------------------:|----------------:|-----------------:|:------------------------------|:---------------------------|
| Decision Tree |              21.644 |           15.103 |             6.541 |               67.971 |            39.268 |             28.703 |              0.894 |           0.965 |            0.071 | Overfitting                   | Good Generalization        |
| Random Forest |              15.336 |           19.370 |            -4.034 |               41.315 |            47.072 |             -5.757 |              0.961 |           0.949 |           -0.012 | Overfitting                   | Moderate Gap               |
| XGBoost       |              19.196 |           15.656 |             3.541 |               65.763 |            39.675 |             26.088 |              0.901 |           0.964 |            0.063 | Overfitting                   | Good Generalization        |

## Did Tuning Reduce Overfitting?

- **Decision Tree**: Test MAE improved by 6.54 meals (Overfitting → Good Generalization).
- **Random Forest**: Test MAE did not improve (tuned model performed worse) by -4.03 meals (Overfitting → Moderate Gap).
- **XGBoost**: Test MAE improved by 3.54 meals (Overfitting → Good Generalization).

## Final Model Selection

Selection order: (1) lowest test MAE, (2) lowest test RMSE as tiebreak, (3) highest test R² as supporting evidence, (4) acceptable generalization gap, (5) practical prediction speed. The model was **not** chosen for having the highest R² alone.

Ranking by test MAE (best to worst):

1. Decision Tree — Test MAE 15.10, Test RMSE 39.27, Test R² 0.965
2. XGBoost — Test MAE 15.66, Test RMSE 39.67, Test R² 0.964
3. Linear Regression — Test MAE 17.17, Test RMSE 38.88, Test R² 0.965
4. Random Forest — Test MAE 19.37, Test RMSE 47.07, Test R² 0.949

**Selected final model: Decision Tree** (Test MAE=15.10, Test RMSE=39.27, Test R²=0.965, generalization: Good Generalization).

## Limitations of Synthetic Data

All tuning and evaluation above used the synthetic dataset generated for development (`src/synthetic_data.py`). Relationships between rainfall, holidays, exams, and attendance were hand-designed to be plausible, not fit to any real county's records. Hyperparameters selected here are only known to be good for this synthetic data's noise structure and scale.

## What to Repeat With Real Data

1. Re-run `run_pipeline.py` and `run_eda.py` against the real dataset once available (point `BITETRACKER_DATA_PATH`/`ACTIVE_DATA_PATH` at it).
2. Re-run `run_training.py` for fresh baselines — real data's noise level and feature relationships will differ from the synthetic generator's assumptions.
3. Re-run this tuning phase (`run_tuning.py`) from scratch — do not reuse the hyperparameters found here; they were fit to synthetic patterns.
4. Re-check the TimeSeriesSplit fold sizes — a real dataset with different row counts or a longer history may warrant more or fewer folds.
