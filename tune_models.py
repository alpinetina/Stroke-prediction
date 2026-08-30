import os
import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import joblib
from train_models import get_feature_target, build_preprocessor, TRAIN_PATH, TEST_PATH
from evaluate_metrics import g_mean_score

RESULTS_PATH = os.path.join("data", "processed", "model_performance_tuned.csv")
MODELS_TUNED_DIR = os.path.join("models_tuned")
N_SPLITS = 5
RANDOM_STATE = 42

SEARCH_SPACE = {
    "LR": (
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=3000,
                            class_weight="balanced", random_state=RANDOM_STATE),
        {"classifier__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
        8,
    ),
    "RF": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1),
        {
            "classifier__max_depth": [4, 6, 8, 10],
            "classifier__min_samples_leaf": [5, 10, 20, 30],
        },
        12,
    ),
    "XGBoost": (
        xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),  # scale_pos_weight set in main()
        {
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__min_child_weight": [1, 5, 10, 20],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__n_estimators": [100, 200, 300],
        },
        20,
    ),
    "LightGBM": (
        lgb.LGBMClassifier(boosting_type="goss", class_weight="balanced",
                            random_state=RANDOM_STATE, verbosity=-1),
        {
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__min_child_samples": [10, 20, 30, 50],
            "classifier__num_leaves": [7, 15, 31],
            "classifier__n_estimators": [100, 200, 300],
        },
        20,
    ),
}

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = get_feature_target(train)
    X_test, y_test = get_feature_target(test)
    preprocessor = build_preprocessor(X_train)

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    SEARCH_SPACE["XGBoost"][0].set_params(scale_pos_weight=scale_pos_weight)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    os.makedirs(MODELS_TUNED_DIR, exist_ok=True)

    for model_name, (base_classifier, param_dist, n_iter) in SEARCH_SPACE.items():
        print(f"\nTuning {model_name} ({n_iter} candidates x {N_SPLITS} folds)")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", base_classifier),
        ])
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=n_iter, scoring="average_precision",
            cv=cv, random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        print(f"Best CV AUC-PRC: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        best_model = search.best_estimator_
        joblib.dump(best_model, os.path.join(MODELS_TUNED_DIR, f"{model_name}_tuned.joblib"))
        test_proba = best_model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.5).astype(int)

        rows.append({
            "model": model_name,
            "cv_AUC-PRC": search.best_score_,
            "test_AUC-ROC": roc_auc_score(y_test, test_proba),
            "test_AUC-PRC": average_precision_score(y_test, test_proba),
            "test_F1": f1_score(y_test, test_pred, zero_division=0),
            "test_G-mean": g_mean_score(y_test, test_pred),
            "best_params": str(search.best_params_),
        })

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 140)
    print("\n" + results.drop(columns=["best_params"]).round(4).to_string(index=False))

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH} (includes best_params per model)")

if __name__ == "__main__":
    main()
