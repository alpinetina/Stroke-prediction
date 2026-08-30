import os
import miceforest as mf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join("data", "processed", "nhanes_merged_raw.csv")
OUT_DIR = os.path.join("data", "processed")

WINSORIZE_COLUMNS = [
    "bmi", "waist_circumference_cm", "systolic_bp", "diastolic_bp",
    "hba1c", "fasting_glucose", "total_cholesterol", "hdl_cholesterol",
    "triglycerides", "omega3_index", "alcohol_drinks_per_week", "rx_med_count",
    "rdw", "mpv", "hematocrit", "hemoglobin", "wbc_count", "platelet_count",
    "neutrophils_count", "lymphocytes_count",
]

BINARY_COLUMNS = [
    "sex", "hypertension_diagnosis", "bp_medication", "diabetes_diagnosis",
    "heart_failure", "coronary_hd", "heart_attack", "liver_disease", "copd",
    "thyroid_disease", "cancer", "smoke_current", "smoke_former", "smoke_never",
    "moderate_activity", "vigorous_activity", "kidney_disease", "rx_med_use",
]
CATEGORICAL_COLUMNS = BINARY_COLUMNS + ["race", "education"]

DESCRIPTIVE_ONLY_COLUMNS = ["vascular_risk_score"]

VASCULAR_RISK_COMPONENTS = [
    "hypertension_diagnosis", "bp_medication", "diabetes_diagnosis",
    "heart_failure", "coronary_hd", "heart_attack", "smoke_current",
    "kidney_disease",
]

MICE_EXCLUDE_COLUMNS = ["SEQN"]

def set_categorical_dtypes(df, columns=CATEGORICAL_COLUMNS):
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def winsorize_continuous(df, columns=WINSORIZE_COLUMNS, lower=0.01, upper=0.99, bounds=None):
    df = df.copy()
    if bounds is None:
        bounds = {c: (df[c].quantile(lower), df[c].quantile(upper)) for c in columns if c in df.columns}
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lower=lo, upper=hi)
    return df, bounds


def impute_missing(df, exclude=MICE_EXCLUDE_COLUMNS, iterations=15, random_state=42, kernel=None):
    impute_cols = [c for c in df.columns if c not in exclude]
    df = df.reset_index(drop=True)

    if kernel is None:
        kernel = mf.ImputationKernel(df[impute_cols], num_datasets=1, random_state=random_state)
        kernel.mice(iterations=iterations)
        completed = kernel.complete_data(dataset=0)
    else:
        completed = kernel.impute_new_data(df[impute_cols]).complete_data(dataset=0)

    for c in exclude:
        if c in df.columns:
            completed[c] = df[c].values
    return completed, kernel

def compare_before_after_imputation(before_df, after_df, columns=WINSORIZE_COLUMNS, out_dir=OUT_DIR):
    """Summary stats + distribution histograms for continuous columns
    before (with missing values) vs after MICE. Saved for thesis figures."""
    fig_dir = os.path.join(out_dir, "imputation_diagnostics")
    os.makedirs(fig_dir, exist_ok=True)

    rows = []
    for c in columns:
        if c not in before_df.columns:
            continue
        rows.append({
            "column": c,
            "missing_before": before_df[c].isna().mean(),
            "mean_before": before_df[c].mean(),
            "std_before": before_df[c].std(),
            "mean_after": after_df[c].mean(),
            "std_after": after_df[c].std(),
        })

        fig, ax = plt.subplots()
        before_df[c].dropna().plot(kind="hist", bins=30, alpha=0.5, density=True,
                                   color="#e64980", label="observed (pre-MICE)", ax=ax)
        after_df[c].plot(kind="hist", bins=30, alpha=0.5, density=True,
                         color="#40c057", label="completed (post-MICE)", ax=ax)
        ax.set_title(c)
        ax.legend()
        fig.savefig(os.path.join(fig_dir, f"{c}_before_after.png"), dpi=100)
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "imputation_comparison.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved histograms {fig_dir}")
    return summary

def compute_engineered_features(df):
    df = df.copy()

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
    df["chol_hdl_ratio"] = df["total_cholesterol"] / df["hdl_cholesterol"].replace(0, np.nan)

    df["nlr"] = df["neutrophils_count"] / df["lymphocytes_count"].replace(0, np.nan)
    df["plr"] = df["platelet_count"] / df["lymphocytes_count"].replace(0, np.nan)

    df["vascular_risk_score"] = df[VASCULAR_RISK_COMPONENTS].astype(int).sum(axis=1)

    return df

if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=DESCRIPTIVE_ONLY_COLUMNS, errors="ignore")
    df = set_categorical_dtypes(df)
    print(f"Loaded {df.shape[0]} rows from {RAW_PATH}")

    train, test = train_test_split(df, test_size=0.20, stratify=df["stroke"], random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_w, bounds = winsorize_continuous(train)
    test_w, _ = winsorize_continuous(test, bounds=bounds)

    bounds_df = pd.DataFrame([
        {"variable": c, "lower_1pct": lo, "upper_99pct": hi}
        for c, (lo, hi) in bounds.items()
    ])
    bounds_path = os.path.join(OUT_DIR, "winsorization_bounds.csv")
    bounds_df.to_csv(bounds_path, index=False)
    print(f"Saved {bounds_path}")

    print("Running MICE imputation")
    train_imputed, kernel = impute_missing(train_w, iterations=15)
    test_imputed, _ = impute_missing(test_w, kernel=kernel)
    compare_before_after_imputation(train_w, train_imputed)

    print("Engineering features")
    train_final = compute_engineered_features(train_imputed)
    test_final = compute_engineered_features(test_imputed)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_out = os.path.join(OUT_DIR, "train_imputed.csv")
    test_out = os.path.join(OUT_DIR, "test_imputed.csv")

    train_final.to_csv(train_out, index=False)
    test_final.to_csv(test_out, index=False)

    print(f"Saved {train_out}")
    print(f"Saved {test_out}")

if __name__ == "__main__":
    main()
