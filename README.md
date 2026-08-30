# Stroke Risk Prediction — NHANES 2021–2023

## Pipeline (run in order)

1. `merge_nhanes.py` — downloads and merges raw NHANES XPT files into one dataset.
2. `impute_and_engineer.py` — cleaning, MICE imputation, feature engineering, train/test split.
3. `train_models.py` — trains baseline LR, RF, XGBoost, LightGBM.
4. `tune_models.py` — hyperparameter tuning via cross-validation.
5. `evaluate_metrics.py` — metric helpers (`g_mean_score`); imported by other scripts, not a required standalone step.
6. `recalibrate.py` — Platt calibration and threshold selection.
7. `run_shap.py` — computes SHAP values for all models.
8. `plot_evaluation.py` — generates all thesis figures and tables.

```bash
python merge_nhanes.py
python impute_and_engineer.py
python train_models.py
python tune_models.py
python recalibrate.py
python run_shap.py
python plot_evaluation.py
```
