import os
import joblib
import warnings
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train_imputed.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_imputed.csv")
MODEL_DIR = os.path.join("models")

TARGET_COL = "stroke"
ID_COL = "SEQN"
DESCRIPTIVE_ONLY_COLUMNS = ["vascular_risk_score"]
CATEGORICAL_COLUMNS = [
    "sex", "race", "education", "hypertension_diagnosis", "bp_medication",
    "diabetes_diagnosis", "heart_failure", "coronary_hd", "heart_attack",
    "liver_disease", "copd", "thyroid_disease", "cancer", "smoke_current",
    "smoke_former", "smoke_never", "moderate_activity", "vigorous_activity",
    "kidney_disease", "rx_med_use",
]

def get_feature_target(df: pd.DataFrame):
    """Split a raw train/test dataframe into (X, y), dropping ID and
    descriptive-only columns."""
    drop_cols = [c for c in [ID_COL] + DESCRIPTIVE_ONLY_COLUMNS if c in df.columns]
    X = df.drop(columns=[TARGET_COL] + drop_cols)
    y = df[TARGET_COL].astype(int)
    return X, y


def load_and_prepare_data():
    """Load train/test sets and split into X and y, dropping non-feature columns."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = get_feature_target(train_df)
    X_test, y_test = get_feature_target(test_df)

    print(f"Loaded Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train: {y_train.sum():.0f}/{len(y_train)} stroke cases ({y_train.mean():.2%})")
    print(f"Test:  {y_test.sum():.0f}/{len(y_test)} stroke cases ({y_test.mean():.2%})")
    print(f"EPV (train stroke cases / features): {y_train.sum() / X_train.shape[1]:.2f}")

    return X_train, y_train, X_test, y_test

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    print(f"Identified {len(num_cols)} continuous features and {len(cat_cols)} categorical/binary features.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        sparse_threshold=0,
    )
    return preprocessor


def get_test_predictions(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return pd.DataFrame({"model": name, "y_true": y_test.values, "y_pred_proba": y_pred_proba})


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    X_train, y_train, X_test, y_test = load_and_prepare_data()
    preprocessor = build_preprocessor(X_train)

    # class weight ratio for XGBoost
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(scale_pos_weight=ratio, n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss")),
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(class_weight="balanced", random_state=42, verbosity=-1)),
        ]),
    }

    # Model selection via 5-fold stratified CV on train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"{name}: CV PR-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nSelected by CV: {best_name} (CV PR-AUC = {cv_scores[best_name]:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)

    fitted = {}
    all_predictions = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        all_predictions.append(get_test_predictions(name, pipeline, X_test, y_test))

    predictions_path = os.path.join(DATA_DIR, "model_predictions.csv")
    pd.concat(all_predictions, ignore_index=True).to_csv(predictions_path, index=False)
    print(f"Saved predictions {predictions_path}")

    best_model = fitted[best_name]

    # Save best performing pipeline artifact
    best_model_path = os.path.join(MODEL_DIR, "best_stroke_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"Best model: '{best_name}' saved to {best_model_path}")

if __name__ == "__main__":
    main()
