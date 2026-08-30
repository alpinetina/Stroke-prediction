import ast
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from evaluate_metrics import g_mean_score
from train_models import TEST_PATH, TRAIN_PATH, build_preprocessor, get_feature_target
from tune_models import RANDOM_STATE, SEARCH_SPACE

TUNED_RESULTS_PATH = os.path.join("data", "processed", "model_performance_tuned.csv")
MODELS_DIR = "models_calibrated"
RESULTS_PATH = os.path.join("data", "processed", "model_performance_calibrated.csv")
EPS = 1e-6

def calibration_in_the_large(y_true, proba):
    p = np.clip(proba, EPS, 1 - EPS)
    logit_p = np.log(p / (1 - p))
    model = sm.Logit(y_true, np.ones(len(y_true)), offset=logit_p).fit(disp=0)
    return model.params[0]

def best_threshold(y_true, proba, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.5, 100)
    scores = [g_mean_score(y_true, (proba >= t).astype(int)) for t in grid]
    return grid[int(np.argmax(scores))]

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    tuned_results = pd.read_csv(TUNED_RESULTS_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = get_feature_target(train)
    X_test, y_test = get_feature_target(test)
    y_test = y_test.values

    preprocessor = build_preprocessor(X_train)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    SEARCH_SPACE["XGBoost"][0].set_params(scale_pos_weight=scale_pos_weight)

    os.makedirs(MODELS_DIR, exist_ok=True)
    rows = []

    for _, row in tuned_results.iterrows():
        model_name = row["model"]
        best_params = ast.literal_eval(row["best_params"])
        base_classifier, _, _ = SEARCH_SPACE[model_name]

        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", base_classifier)])
        pipe.set_params(**best_params)

        pipe.fit(X_train, y_train)
        proba_before = pipe.predict_proba(X_test)[:, 1]
        brier_before = brier_score_loss(y_test, proba_before)
        citl_before = calibration_in_the_large(y_test, proba_before)

        calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof_proba = cross_val_predict(
            calibrated, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        threshold = best_threshold(y_train.values, oof_proba)

        calibrated.fit(X_train, y_train)

        joblib.dump(calibrated, os.path.join(MODELS_DIR, f"{model_name}_calibrated.joblib"))

        proba = calibrated.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        rows.append({
            "model": model_name,
            "brier_before": brier_before,
            "brier_after": brier_score_loss(y_test, proba),
            "citl_before": citl_before,
            "citl_after": calibration_in_the_large(y_test, proba),
            "AUC-ROC": roc_auc_score(y_test, proba),
            "AUC-PRC": average_precision_score(y_test, proba),
            "threshold_used": threshold,
            "F1": f1_score(y_test, pred, zero_division=0),
            "G-mean": g_mean_score(y_test, pred),
            "mean_predicted_proba": proba.mean(),
        })
        print(f"Calibrated and saved {model_name} (threshold={threshold:.3f})")

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n" + results.round(4).to_string(index=False))
    print(f"\nActual test-set stroke rate: {y_test.mean():.4f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH}")
    print(f"Saved 4 calibrated models {MODELS_DIR}/")

if __name__ == "__main__":
    main()
