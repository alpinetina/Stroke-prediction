import os
import warnings
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(DATA_DIR, "train_imputed.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_imputed.csv")

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
    # splits raw train/test dataframe into (X, y), dropping ID and descriptive-only columns
    drop_cols = [c for c in [ID_COL] + DESCRIPTIVE_ONLY_COLUMNS if c in df.columns]
    X = df.drop(columns=[TARGET_COL] + drop_cols)
    y = df[TARGET_COL].astype(int)
    return X, y


def load_and_prepare_data():
    #loads train/test sets and split into X and y, dropping non-feature columns
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


def main():
    #reports dataset shapes and EPV (section 5.1)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    X_train, y_train, X_test, y_test = load_and_prepare_data()
    build_preprocessor(X_train)

if __name__ == "__main__":
    main()
