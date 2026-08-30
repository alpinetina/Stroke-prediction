import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

PREDICTIONS_PATH = os.path.join("data", "processed", "model_predictions.csv")
OUTPUT_PATH = os.path.join("data", "processed", "model_performance_comparison.csv")
DEFAULT_THRESHOLD = 0.5

def g_mean_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return np.sqrt(sensitivity * specificity)


def compute_metrics(predictions, threshold=DEFAULT_THRESHOLD):
    rows = []

    for model, group in predictions.groupby("model"):
        y_true = group["y_true"].values
        y_proba = group["y_pred_proba"].values
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "model": model,
            "AUC-ROC": roc_auc_score(y_true, y_proba),
            "AUC-PRC": average_precision_score(y_true, y_proba),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "G-mean": g_mean_score(y_true, y_pred),
        })

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def main():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Predictions file not found at '{PREDICTIONS_PATH}'. "
            "Run 'train_models.py' first to generate model predictions."
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    n_models = predictions["model"].nunique()

    print(f"Loaded {predictions.shape[0]} prediction records ({n_models} models)")

    results = compute_metrics(predictions)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    print("\n Model Performance Comparison")
    print(results.round(4).to_string(index=False))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPerformance comparison saved {OUTPUT_PATH}")

    best_row = results.loc[results["AUC-PRC"].idxmax()]
    print(f"\nHighest AUC-PRC: {best_row['model']} ({best_row['AUC-PRC']:.4f})")

if __name__ == "__main__":
    main()
