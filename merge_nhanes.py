import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = os.path.join("data", "processed", "nhanes_merged_raw.csv")

# Descriptive-only columns (Table 1 characteristics)
DESCRIPTIVE_ONLY_COLUMNS = ["vascular_risk_score"]

def load_xpt(name: str, columns: list) -> pd.DataFrame:
    """Load an XPT file, raising if the file or any requested column
    is missing."""
    file_path = os.path.join(RAW_DIR, f"{name}.xpt")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    df = pd.read_sas(file_path, format="xport")

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in {name}.xpt: {missing_cols}")

    return df[columns].copy()


def recode_missing(series: pd.Series, missing_codes: list) -> pd.Series:
    """Convert NHANES Refused/Don't know sentinel codes to NaN."""
    return series.replace(missing_codes, np.nan)


def compute_alcohol(alq: pd.DataFrame) -> pd.DataFrame:
    """Drinks/week = frequency(ALQ121) x quantity(ALQ130). ALQ111==2
    (never drank) sets 0 directly, since ALQ121/ALQ130 are skipped
    for those respondents."""
    alq = alq.copy()
    alq["ALQ111"] = recode_missing(alq["ALQ111"], [7, 9])
    alq["ALQ121"] = recode_missing(alq["ALQ121"], [77, 99])
    alq["ALQ130"] = recode_missing(alq["ALQ130"], [777, 999])

    ALQ_FREQ_PER_WEEK = {
        0: 0.0,  # never in the last year
        1: 7.0,  # every day
        2: 6.0,  # nearly every day
        3: 3.5,  # 3-4 times a week
        4: 2.0,  # 2 times a week
        5: 1.0,  # once a week
        6: 0.62,  # 2-3 times a month
        7: 0.23,  # once a month
        8: 0.173,  # 7-11 times a year
        9: 0.087,  # 3-6 times a year
        10: 0.029,  # 1-2 times a year
    }

    freq_per_week = alq["ALQ121"].map(ALQ_FREQ_PER_WEEK)
    drinks_per_week = freq_per_week * alq["ALQ130"]
    drinks_per_week = np.where(alq["ALQ121"] == 0, 0.0, drinks_per_week)
    drinks_per_week = np.where(alq["ALQ111"] == 2, 0.0, drinks_per_week)

    alq["alcohol_drinks_per_week"] = drinks_per_week
    return alq[["SEQN", "alcohol_drinks_per_week"]]


def load_prescriptions() -> pd.DataFrame:
    """RXQ_RX_L is drug-level (multiple rows per SEQN); RXQ033/RXQ050
    are person-level and repeated on every row, so keep one row per
    person. rx_med_count is set to 0 where rx_med_use==0 and count is
    NaN (known zero, not missing)."""
    rxq = load_xpt("RXQ_RX_L", ["SEQN", "RXQ033", "RXQ050"])
    rxq["RXQ033"] = recode_missing(rxq["RXQ033"], [7, 9])
    rxq["RXQ050"] = recode_missing(rxq["RXQ050"], [7, 9])
    rxq = rxq.drop_duplicates(subset="SEQN", keep="first")
    rxq = rxq.rename(columns={"RXQ050": "rx_med_count"})
    rxq["rx_med_use"] = (rxq["RXQ033"] == 1).astype("Int64")
    rxq.loc[
        (rxq["rx_med_use"] == 0) & (rxq["rx_med_count"].isna()),
        "rx_med_count",
    ] = 0
    return rxq[["SEQN", "rx_med_use", "rx_med_count"]]


def main():
    demo = load_xpt("DEMO_L", ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3",
                                "DMDEDUC2", "INDFMPIR"])
    demo = demo.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",
        "RIDRETH3": "race",
        "DMDEDUC2": "education",
        "INDFMPIR": "income_poverty_ratio",
    })
    demo["education"] = recode_missing(demo["education"], [7, 9])

    bpxo = load_xpt("BPXO_L", ["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3",
                                "BPXODI1", "BPXODI2", "BPXODI3"])
    bpxo["systolic_bp"] = bpxo[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    bpxo["diastolic_bp"] = bpxo[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    bpxo = bpxo[["SEQN", "systolic_bp", "diastolic_bp"]]

    bmx = load_xpt("BMX_L", ["SEQN", "BMXBMI", "BMXWAIST"]).rename(columns={
        "BMXBMI": "bmi", "BMXWAIST": "waist_circumference_cm",
    })

    cbc = load_xpt("CBC_L", ["SEQN", "LBXRDW", "LBXMPSI", "LBXHCT", "LBXHGB",
                             "LBXWBCSI", "LBXPLTSI", "LBDNENO", "LBDLYMNO"])
    cbc = cbc.rename(columns={
        "LBXRDW": "rdw",
        "LBXMPSI": "mpv",
        "LBXHCT": "hematocrit",
        "LBXHGB": "hemoglobin",
        "LBXWBCSI": "wbc_count",
        "LBXPLTSI": "platelet_count",
        "LBDNENO": "neutrophils_count",
        "LBDLYMNO": "lymphocytes_count",
    })

    tchol = load_xpt("TCHOL_L", ["SEQN", "LBXTC"]).rename(
        columns={"LBXTC": "total_cholesterol"})

    hdl = load_xpt("HDL_L", ["SEQN", "LBDHDD"]).rename(
        columns={"LBDHDD": "hdl_cholesterol"})

    trigly = load_xpt("TRIGLY_L", ["SEQN", "LBXTLG", "LBDLDL"]).rename(
        columns={"LBXTLG": "triglycerides", "LBDLDL": "ldl_cholesterol"})

    ghb = load_xpt("GHB_L", ["SEQN", "LBXGH"]).rename(columns={"LBXGH": "hba1c"})

    glu = load_xpt("GLU_L", ["SEQN", "LBXGLU"]).rename(columns={"LBXGLU": "fasting_glucose"})

    # omega3_index: EPA + DHA (LBXPPE + LBXPHA), the standard Omega-3 Index, used instead of all 21 individual FAR_L fatty acid columns
    far = load_xpt("FAR_L", ["SEQN", "LBXPPE", "LBXPHA"])
    far["omega3_index"] = far["LBXPPE"] + far["LBXPHA"]
    far = far[["SEQN", "omega3_index"]]

    mcq = load_xpt("MCQ_L", ["SEQN", "MCQ160F", "MCQ160B", "MCQ160C", "MCQ160E",
                              "MCQ160L", "MCQ160P", "MCQ160M", "MCQ220"])
    mcq = mcq.rename(columns={
        "MCQ160F": "stroke", "MCQ160B": "heart_failure",
        "MCQ160C": "coronary_hd", "MCQ160E": "heart_attack",
        "MCQ160L": "liver_disease", "MCQ160P": "copd",
        "MCQ160M": "thyroid_disease", "MCQ220": "cancer",
    })
    for col in ["heart_failure", "coronary_hd", "heart_attack",
                "liver_disease", "copd", "thyroid_disease", "cancer"]:
        mcq[col] = recode_missing(mcq[col], [7, 9])
        mcq[col] = (mcq[col] == 1).astype("Int64")
    # stroke stays raw here; cleaned during inclusion/exclusion below.

    bpq = load_xpt("BPQ_L", ["SEQN", "BPQ020", "BPQ150"]).rename(columns={
        "BPQ020": "hypertension_diagnosis", "BPQ150": "bp_medication",
    })
    for col in ["hypertension_diagnosis", "bp_medication"]:
        bpq[col] = recode_missing(bpq[col], [7, 9])
        bpq[col] = (bpq[col] == 1).astype("Int64")

    diq = load_xpt("DIQ_L", ["SEQN", "DIQ010"]).rename(columns={"DIQ010": "diabetes_diagnosis"})
    diq["diabetes_diagnosis"] = recode_missing(diq["diabetes_diagnosis"], [7, 9])
    diq["diabetes_diagnosis"] = (diq["diabetes_diagnosis"] == 1).astype("Int64")

    smq = load_xpt("SMQ_L", ["SEQN", "SMQ020", "SMQ040"])
    smq["SMQ020"] = recode_missing(smq["SMQ020"], [7, 9])
    smq["SMQ040"] = recode_missing(smq["SMQ040"], [7, 9])
    smq["smoke_never"] = (smq["SMQ020"] == 2).astype("Int64")
    smq["smoke_current"] = ((smq["SMQ020"] == 1) & (smq["SMQ040"].isin([1, 2]))).astype("Int64")
    smq["smoke_former"] = ((smq["SMQ020"] == 1) & (smq["SMQ040"] == 3)).astype("Int64")
    smq = smq[["SEQN", "smoke_current", "smoke_former", "smoke_never"]]

    alq_raw = load_xpt("ALQ_L", ["SEQN", "ALQ111", "ALQ121", "ALQ130"])
    alq = compute_alcohol(alq_raw)

    paq = load_xpt("PAQ_L", ["SEQN", "PAD790Q", "PAD810Q"])
    paq["PAD790Q"] = recode_missing(paq["PAD790Q"], [7777, 9999])
    paq["PAD810Q"] = recode_missing(paq["PAD810Q"], [7777, 9999])
    paq["moderate_activity"] = ((paq["PAD790Q"].notna()) & (paq["PAD790Q"] > 0)).astype("Int64")
    paq["vigorous_activity"] = ((paq["PAD810Q"].notna()) & (paq["PAD810Q"] > 0)).astype("Int64")
    paq = paq[["SEQN", "moderate_activity", "vigorous_activity"]]

    kiq = load_xpt("KIQ_U_L", ["SEQN", "KIQ022"]).rename(columns={"KIQ022": "kidney_disease"})
    kiq["kidney_disease"] = recode_missing(kiq["kidney_disease"], [7, 9])
    kiq["kidney_disease"] = (kiq["kidney_disease"] == 1).astype("Int64")

    rxq = load_prescriptions()

    frames = [demo, bpxo, bmx, cbc, tchol, hdl, trigly, ghb, glu, far,
              mcq, bpq, diq, smq, alq, paq, kiq, rxq]
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="SEQN", how="left")

    # Descriptive summary of comorbidity count (Table 1 only)
    risk_components = [
        "hypertension_diagnosis", "bp_medication", "diabetes_diagnosis",
        "heart_failure", "coronary_hd", "heart_attack", "smoke_current",
        "kidney_disease",
    ]
    merged["vascular_risk_score"] = merged[risk_components].sum(axis=1, min_count=1)

    n_before = len(merged)

    merged = merged[merged["age"] >= 20]
    n_after_age = len(merged)

    merged["stroke"] = recode_missing(merged["stroke"], [7, 9])
    merged = merged.dropna(subset=["stroke"])
    n_after_stroke = len(merged)

    merged["stroke"] = (merged["stroke"] == 1).astype(int)
    merged = merged.dropna(subset=["age", "sex", "race", "income_poverty_ratio"])
    n_final = len(merged)
    n_stroke_final = int(merged["stroke"].sum())

    flow = pd.DataFrame([
        {"stage": "merged (all ages)", "n": n_before},
        {"stage": "age >= 20", "n": n_after_age},
        {"stage": "valid stroke response", "n": n_after_stroke},
        {"stage": "complete core covariates (final cohort)", "n": n_final},
    ])
    flow_path = os.path.join("data", "processed", "sample_selection_flow.csv")
    flow.to_csv(flow_path, index=False)
    print(f"Saved {flow_path}")
    print(f"Final dataset: {n_final} records, {n_stroke_final} stroke cases ({n_stroke_final/n_final:.2%})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved {merged.shape[0]} rows, {merged.shape[1]} columns {os.path.abspath(OUT_PATH)}")

if __name__ == "__main__":
    main()
