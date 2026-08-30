import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from train_models import get_feature_target, TRAIN_PATH, TEST_PATH
from shap.plots._style import set_style

FIGURES_DIR = "figures"
MODELS_TUNED_DIR = "models_tuned"
MODELS_CALIBRATED_DIR = "models_calibrated"
RAW_PATH = os.path.join("data", "processed", "nhanes_merged_raw.csv")
CALIBRATED_RESULTS_PATH = os.path.join("data", "processed", "model_performance_calibrated.csv")
SHAP_VALUES_DIR = "shap_values"

TABLE1_PATH = os.path.join("data", "processed", "table1_cohort_characteristics.csv")
CORR_PEARSON_PATH = os.path.join("data", "processed", "correlation_pearson.csv")
CONFUSION_STATS_PATH = os.path.join("data", "processed", "confusion_matrix_stats.csv")
DCA_RESULTS_PATH = os.path.join("data", "processed", "decision_curve_analysis.csv")
PREDICTORS_NATIVE_PATH = os.path.join("data", "processed", "consistent_top_predictors.csv")
PREDICTORS_SHAP_PATH = os.path.join("data", "processed", "consistent_top_predictors_shap.csv")

MODEL_ORDER = ["LR", "RF", "XGBoost", "LightGBM"]
TREE_MODELS = ["RF", "XGBoost", "LightGBM"]
SHAP_FEATURED_MODEL = "LR"

COLORS = {"LR": "#c2255c", "RF": "#2b8a3e", "XGBoost": "#e64980", "LightGBM": "#40c057"}
DIVERGING_CMAP = "PiYG"
# Colors for stubborn SHAP plots
_piyg = plt.get_cmap(DIVERGING_CMAP)
pink_hex = mcolors.to_hex(_piyg(0.2))
green_hex = mcolors.to_hex(_piyg(0.8))

set_style(
    primary_color_positive=green_hex,
    primary_color_negative=pink_hex,
    secondary_color_positive=green_hex,
    secondary_color_negative=pink_hex,
)

EPS = 1e-6
TOP_N = 10

RACE_LABELS = {
    "race_2.0": "Race: Other Hispanic",
    "race_3.0": "Race: Non-Hispanic White",
    "race_4.0": "Race: Non-Hispanic Black",
    "race_6.0": "Race: Non-Hispanic Asian",
    "race_7.0": "Race: Other/Multiracial",
}

def display_name(col):
    return RACE_LABELS.get(col, col)


# Table 1
CONTINUOUS_VARS = [
    "age", "income_poverty_ratio", "bmi", "waist_circumference_cm",
    "systolic_bp", "diastolic_bp", "hba1c", "fasting_glucose",
    "total_cholesterol", "hdl_cholesterol", "triglycerides", "omega3_index",
    "alcohol_drinks_per_week", "rx_med_count", "rdw", "mpv", "hematocrit",
    "hemoglobin", "wbc_count", "platelet_count", "neutrophils_count",
    "lymphocytes_count", "vascular_risk_score",
]
CATEGORICAL_VARS = [
    "sex", "race", "education", "hypertension_diagnosis", "bp_medication",
    "diabetes_diagnosis", "heart_failure", "coronary_hd", "heart_attack",
    "liver_disease", "copd", "thyroid_disease", "cancer", "smoke_current",
    "smoke_former", "smoke_never", "moderate_activity", "vigorous_activity",
    "kidney_disease", "rx_med_use",
]


def _continuous_row(df, col, group_col="stroke"):
    g0 = df.loc[df[group_col] == 0, col].dropna()
    g1 = df.loc[df[group_col] == 1, col].dropna()
    pval = np.nan
    if len(g0) >= 2 and len(g1) >= 2:
        _, pval = stats.ttest_ind(g0, g1, equal_var=False)
    return {
        "variable": col, "type": "continuous",
        "no_stroke": f"{g0.mean():.2f} ({g0.std():.2f})" if len(g0) else "-",
        "stroke": f"{g1.mean():.2f} ({g1.std():.2f})" if len(g1) else "-",
        "p_value": pval, "n_missing": int(df[col].isna().sum()),
    }


def _categorical_rows(df, col, group_col="stroke"):
    sub = df[[col, group_col]].dropna(subset=[col])
    n_missing = int(df[col].isna().sum())
    table = pd.crosstab(sub[col], sub[group_col])

    pval = np.nan
    if table.shape == (2, 2):
        _, pval = stats.fisher_exact(table.values)
    elif table.size > 0:
        _, pval, _, _ = stats.chi2_contingency(table)

    rows = []
    n0_total = table[0].sum() if 0 in table.columns else 0
    n1_total = table[1].sum() if 1 in table.columns else 0
    for level in table.index:
        n0 = table.loc[level, 0] if 0 in table.columns else 0
        n1 = table.loc[level, 1] if 1 in table.columns else 0
        rows.append({
            "variable": f"{col} = {level}", "type": "categorical",
            "no_stroke": f"{n0} ({n0 / n0_total * 100 if n0_total else 0:.1f}%)",
            "stroke": f"{n1} ({n1 / n1_total * 100 if n1_total else 0:.1f}%)",
            "p_value": pval, "n_missing": n_missing,
        })
    return rows


def _build_table1(df, label):
    rows = [{
        "variable": "N", "type": "n",
        "no_stroke": str(int((df["stroke"] == 0).sum())),
        "stroke": str(int((df["stroke"] == 1).sum())),
        "p_value": np.nan, "n_missing": 0,
    }]
    for c in CONTINUOUS_VARS:
        if c in df.columns:
            rows.append(_continuous_row(df, c))
    for c in CATEGORICAL_VARS:
        if c in df.columns:
            rows.extend(_categorical_rows(df, c))
    table = pd.DataFrame(rows)
    table.insert(0, "cohort", label)
    return table


def run_table1(train_df, test_df):
    raw = pd.read_csv(RAW_PATH)
    imputed = pd.concat([train_df, test_df], ignore_index=True)
    combined = pd.concat(
        [_build_table1(raw, "raw (pre-imputation)"), _build_table1(imputed, "MICE-completed")],
        ignore_index=True,
    )
    combined.to_csv(TABLE1_PATH, index=False)


# Correlation heatmap
CORRELATION_VARS = [
    "stroke", "age", "income_poverty_ratio", "bmi", "waist_circumference_cm",
    "systolic_bp", "diastolic_bp", "pulse_pressure", "map", "hba1c",
    "fasting_glucose", "total_cholesterol", "hdl_cholesterol", "triglycerides",
    "chol_hdl_ratio", "omega3_index", "alcohol_drinks_per_week", "rx_med_count",
    "rdw", "mpv", "hematocrit", "hemoglobin", "wbc_count", "platelet_count",
    "neutrophils_count", "lymphocytes_count", "nlr", "plr", "vascular_risk_score",
]

def run_correlation_heatmap(imputed_df):
    cols = [c for c in CORRELATION_VARS if c in imputed_df.columns]
    corr = imputed_df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=6)
    ax.set_title("Pearson Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_png = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(CORR_PEARSON_PATH)


# Discrimination + calibration
def load_calibrated_probas(X_test):
    probas = {}
    for model_name in MODEL_ORDER:
        model = joblib.load(os.path.join(MODELS_CALIBRATED_DIR, f"{model_name}_calibrated.joblib"))
        probas[model_name] = model.predict_proba(X_test)[:, 1]
    return probas


def run_discrimination(probas, y_test):
    prevalence = y_test.mean()
    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(11, 5))

    for model_name in MODEL_ORDER:
        proba = probas[model_name]
        color = COLORS[model_name]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.8,
                    label=f"{model_name} (AUC={roc_auc_score(y_test, proba):.3f})")
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax_prc.plot(recall, precision, color=color, linewidth=1.8,
                    label=f"{model_name} (AUC={average_precision_score(y_test, proba):.3f})")

    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Chance")
    ax_roc.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curve", xlim=(0, 1), ylim=(0, 1.02))
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.25)

    ax_prc.axhline(prevalence, color="grey", linestyle="--", linewidth=1, label=f"Chance ({prevalence:.3f})")
    ax_prc.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve",
               xlim=(0, 1), ylim=(0, 1.02))
    ax_prc.legend(loc="upper right", fontsize=9)
    ax_prc.grid(alpha=0.25)

    fig.suptitle("Model Discrimination - Calibrated Models", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(FIGURES_DIR, "discrimination_roc_prc.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Confusion matrices
def _plot_cm(ax, cm, model_name, threshold):
    ax.imshow(cm, cmap=DIVERGING_CMAP)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Stroke", "Stroke"])
    ax.set_yticklabels(["No Stroke", "Stroke"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} (threshold={threshold:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=22, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")


def run_confusion_matrices(probas, y_test):
    thresholds = pd.read_csv(CALIBRATED_RESULTS_PATH).set_index("model")["threshold_used"]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        threshold = thresholds[model_name]
        pred = (probas[model_name] >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        _plot_cm(ax, cm, model_name, threshold)

        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "model": model_name, "threshold": threshold,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        })

    fig.suptitle("Confusion Matrices - Calibrated Models", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(CONFUSION_STATS_PATH, index=False)

# Decision curve
DCA_THRESHOLDS = np.arange(0.01, 0.40, 0.01)


def _net_benefit_model(y_true, y_proba, thresholds):
    n = len(y_true)
    nb = np.zeros(len(thresholds))
    for i, pt in enumerate(thresholds):
        pred_pos = y_proba >= pt
        tp = np.sum(pred_pos & (y_true == 1))
        fp = np.sum(pred_pos & (y_true == 0))
        nb[i] = tp / n - fp / n * (pt / (1 - pt))
    return nb


def _net_benefit_treat_all(y_true, thresholds):
    n = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    return np.array([tp / n - fp / n * (pt / (1 - pt)) for pt in thresholds])


def run_dca(probas, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    nb_all = _net_benefit_treat_all(y_test, DCA_THRESHOLDS)
    ax.plot(DCA_THRESHOLDS, nb_all, color="black", linewidth=1.3, label="Treat all")
    ax.plot(DCA_THRESHOLDS, np.zeros(len(DCA_THRESHOLDS)), color="grey", linestyle="--",
            linewidth=1.3, label="Treat none")

    rows = [{"threshold": pt, "strategy": "treat_all", "net_benefit": nb} for pt, nb in zip(DCA_THRESHOLDS, nb_all)]
    rows += [{"threshold": pt, "strategy": "treat_none", "net_benefit": 0.0} for pt in DCA_THRESHOLDS]

    nb_models = {}
    for model_name in MODEL_ORDER:
        nb_model = _net_benefit_model(y_test, probas[model_name], DCA_THRESHOLDS)
        nb_models[model_name] = nb_model
        ax.plot(DCA_THRESHOLDS, nb_model, color=COLORS[model_name], linewidth=1.8, label=model_name)
        rows += [{"threshold": pt, "strategy": model_name, "net_benefit": nb}
                 for pt, nb in zip(DCA_THRESHOLDS, nb_model)]

    all_model_values = np.concatenate(list(nb_models.values()))
    ax.set_ylim(min(-0.01, all_model_values.min() * 1.2), all_model_values.max() * 1.2)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlim(DCA_THRESHOLDS[0], DCA_THRESHOLDS[-1])
    ax.set_xlabel("Threshold probability (Pt)")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis - Calibrated Models")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_png = os.path.join(FIGURES_DIR, "decision_curve_analysis.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(DCA_RESULTS_PATH, index=False)

# Consistent top predictors
def _get_native_importance(model_name):
    pipe = joblib.load(os.path.join(MODELS_TUNED_DIR, f"{model_name}_tuned.joblib"))
    preprocessor = pipe.named_steps["preprocessor"]
    classifier = pipe.named_steps["classifier"]
    feature_names = [n.split("__")[-1] for n in preprocessor.get_feature_names_out()]

    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        df = pd.DataFrame({"feature": feature_names, "value": values})
        df["abs_value"] = df["value"].abs()
        df["direction"] = np.where(df["value"] > 0, "+", "-")
    else:
        values = classifier.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "value": values})
        df["abs_value"] = df["value"]
        df["direction"] = ""

    df = df.sort_values("abs_value", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["model"] = model_name
    return df


def _get_shap_importance(model_name):
    d = np.load(os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap.npz"), allow_pickle=True)
    feature_names = list(d["feature_names"])
    shap_values = d["shap_values"]
    df = pd.DataFrame({"feature": feature_names, "value": np.abs(shap_values).mean(axis=0)})
    df["abs_value"] = df["value"]
    df["direction"] = np.where(shap_values.mean(axis=0) > 0, "+", "-")
    df = df.sort_values("abs_value", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["model"] = model_name
    return df


def _build_consistency_table(get_table_fn):
    all_top, top_sets = [], {}
    for model_name in MODEL_ORDER:
        top = get_table_fn(model_name).head(TOP_N)
        all_top.append(top)
        top_sets[model_name] = set(top["feature"])
    all_top = pd.concat(all_top, ignore_index=True)

    rows = []
    for feat in sorted(set().union(*top_sets.values())):
        n_models = sum(feat in top_sets[m] for m in MODEL_ORDER)
        row = {"feature": feat, "n_models_top10": n_models}
        for m in MODEL_ORDER:
            rank = None
            if feat in top_sets[m]:
                rank = int(all_top[(all_top["model"] == m) & (all_top["feature"] == feat)]["rank"].iloc[0])
            row[f"{m}_rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_models_top10", "feature"], ascending=[False, True]).reset_index(drop=True)


def run_consistent_predictors():
    native = _build_consistency_table(_get_native_importance)
    native.to_csv(PREDICTORS_NATIVE_PATH, index=False)

    shap_based = _build_consistency_table(_get_shap_importance)
    shap_based.to_csv(PREDICTORS_SHAP_PATH, index=False)

# SHAP
INTERACTION_FOR_HTN = "bp_medication_1"
SHAP_MAX_DISPLAY = 15
LOCAL_MAX_DISPLAY = 12


def _load_explanation(model_name):
    d = np.load(os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap.npz"), allow_pickle=True)
    feature_names = list(d["feature_names"])
    expl = shap.Explanation(
        values=d["shap_values"], base_values=float(d["expected_value"]),
        data=d["feature_values"], feature_names=feature_names,
    )
    return expl, feature_names


def run_shap_summary():
    expl, feature_names = _load_explanation(SHAP_FEATURED_MODEL)
    expl.feature_names = [display_name(c) for c in feature_names]

    plt.figure()
    shap.plots.beeswarm(expl, max_display=SHAP_MAX_DISPLAY, color=DIVERGING_CMAP, show=False)
    plt.title(f"SHAP Summary - {SHAP_FEATURED_MODEL}")
    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, f"shap_summary_{SHAP_FEATURED_MODEL}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def run_shap_dependence():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    cmap_obj = plt.get_cmap(DIVERGING_CMAP)

    for col, model_name in enumerate(TREE_MODELS):
        expl, feature_names = _load_explanation(model_name)

        plt.sca(axes[0, col])
        shap.plots.scatter(expl[:, "age"], color=expl, cmap=cmap_obj, ax=axes[0, col], show=False)
        axes[0, col].set_title(f"{model_name}: age")

        plt.sca(axes[1, col])
        if INTERACTION_FOR_HTN in feature_names:
            shap.plots.scatter(expl[:, "hypertension_diagnosis_1"], color=expl[:, INTERACTION_FOR_HTN],
                               cmap=cmap_obj, ax=axes[1, col], show=False)
        else:
            shap.plots.scatter(expl[:, "hypertension_diagnosis_1"], cmap=cmap_obj, ax=axes[1, col], show=False)
        axes[1, col].set_title(f"{model_name}: hypertension_diagnosis")

    fig.suptitle("SHAP Dependence - Tuned Models", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(FIGURES_DIR, "shap_dependence.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


import matplotlib.colors as mcolors


def _plot_local(patient_idx, label, filename_stub):
    expl, _ = _load_explanation(SHAP_FEATURED_MODEL)
    plt.figure()

    shap.plots.waterfall(expl[patient_idx], max_display=LOCAL_MAX_DISPLAY, show=False)

    plt.title(f"{label} - {SHAP_FEATURED_MODEL} (test row {patient_idx})")
    out_png = os.path.join(FIGURES_DIR, f"{filename_stub}_{SHAP_FEATURED_MODEL}.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def run_shap_local(X_test, y_test):
    selection_pipe = joblib.load(os.path.join(MODELS_TUNED_DIR, f"{SHAP_FEATURED_MODEL}_tuned.joblib"))
    proba = selection_pipe.predict_proba(X_test)[:, 1]
    high_idx = int(np.argmax(proba))
    low_idx = int(np.argmin(proba))
    _plot_local(high_idx, "Highest-Risk Patient", "shap_local_high_risk")
    _plot_local(low_idx, "Lowest-Risk Patient", "shap_local_low_risk")


def main():
    warnings.filterwarnings("ignore")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_test, y_test = get_feature_target(test)

    run_table1(train, test)

    run_correlation_heatmap(pd.concat([train, test], ignore_index=True))

    probas = load_calibrated_probas(X_test)
    run_discrimination(probas, y_test.values)

    run_confusion_matrices(probas, y_test.values)

    run_dca(probas, y_test.values)

    run_consistent_predictors()

    run_shap_summary()
    run_shap_dependence()
    run_shap_local(X_test, y_test)

    print("Saved all generated CSV files in data\\processed\\ and plots in figures\\")

if __name__ == "__main__":
    main()
