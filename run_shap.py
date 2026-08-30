import os
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from train_models import get_feature_target, TRAIN_PATH, TEST_PATH
import shap.explainers._tree as _shap_tree_module

MODELS_DIR = "models_tuned"
SHAP_VALUES_DIR = "shap_values"
TREE_MODELS = ["RF", "XGBoost", "LightGBM"]
LINEAR_MODELS = ["LR"]
MODEL_ORDER = LINEAR_MODELS + TREE_MODELS

_original_decode_ubjson = _shap_tree_module.decode_ubjson_buffer
_shap_tree_module.decode_ubjson_buffer = _patched_decode_ubjson

def _patched_decode_ubjson(fd):
 jmodel = _original_decode_ubjson(fd)
 try:
  param = jmodel["learner"]["learner_model_param"]
  bs = param.get("base_score")
  if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
   param["base_score"] = bs[1:-1]
 except (KeyError, TypeError):
  pass
 return jmodel

def get_positive_class_shap(explainer, X):
 shap_values = explainer.shap_values(X)
 expected_value = explainer.expected_value

 if isinstance(shap_values, list):
  shap_values = shap_values[1]
  expected_value = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
 elif shap_values.ndim == 3:
  shap_values = shap_values[:, :, 1]
  expected_value = expected_value[1] if hasattr(expected_value, "__len__") else expected_value

 return shap_values, expected_value


def main():
 warnings.filterwarnings("ignore")

 train = pd.read_csv(TRAIN_PATH)
 test = pd.read_csv(TEST_PATH)
 X_train, _ = get_feature_target(train)
 X_test, y_test = get_feature_target(test)

 os.makedirs(SHAP_VALUES_DIR, exist_ok=True)

 for model_name in MODEL_ORDER:
  print(f"Computing SHAP values for {model_name}")
  pipe = joblib.load(os.path.join(MODELS_DIR, f"{model_name}_tuned.joblib"))
  preprocessor = pipe.named_steps["preprocessor"]
  classifier = pipe.named_steps["classifier"]

  X_test_transformed = preprocessor.transform(X_test)
  raw_names = preprocessor.get_feature_names_out()
  feature_names = [n.split("__")[-1] for n in raw_names]

  if model_name in TREE_MODELS:
   explainer = shap.TreeExplainer(classifier)
   shap_values, expected_value = get_positive_class_shap(explainer, X_test_transformed)
  else:
   X_train_transformed = preprocessor.transform(X_train)
   explainer = shap.LinearExplainer(classifier, X_train_transformed)
   shap_values = explainer.shap_values(X_test_transformed)
   expected_value = explainer.expected_value
   if hasattr(expected_value, "__len__"):
    expected_value = expected_value[0]

  np.savez(
   os.path.join(SHAP_VALUES_DIR, f"{model_name}_shap.npz"),
   shap_values=shap_values,
   expected_value=expected_value,
   feature_values=X_test_transformed,
   feature_names=np.array(feature_names, dtype=object),
  )
  print(f"Saved {SHAP_VALUES_DIR}/{model_name}_shap.npz")

if __name__ == "__main__":
 main()
