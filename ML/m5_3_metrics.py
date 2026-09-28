import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT /
    "Data" /
    "placement_predict_cleaned.csv"
)

PLOT_DIR = ROOT / "ML" / "plots"
MODEL_DIR = ROOT / "ML" / "models"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=== M5-3: Classification + Regression Metrics ===")
print(f"Dataset shape: {df.shape}")


# ============================================================
# COMMON FEATURES
# ============================================================

drop_cols = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]


# ============================================================
# CLASSIFICATION
# ============================================================

classification_drop = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
    "PlacementStatus",
    "Salary Package",
]

classification_features = [
    c for c in df.columns
    if c not in classification_drop
]

X_class = df[
    classification_features
].copy()

y_class = df[
    "PlacementStatus"
].copy()


# ------------------------------------------------------------
# Classification preprocessing
# ------------------------------------------------------------

class_num_features = (
    X_class
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

class_cat_features = (
    X_class
    .select_dtypes(exclude=[np.number])
    .columns
    .tolist()
)


class_num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


class_cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


class_preprocessor = ColumnTransformer([
    (
        "num",
        class_num_pipeline,
        class_num_features
    ),
    (
        "cat",
        class_cat_pipeline,
        class_cat_features
    )
])


classification_pipeline = Pipeline([
    (
        "preprocessor",
        class_preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


# ------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------

X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_class
)


# ------------------------------------------------------------
# Train classification model
# ------------------------------------------------------------

classification_pipeline.fit(
    X_train_c,
    y_train_c
)


y_pred_class = (
    classification_pipeline
    .predict(X_val_c)
)

y_prob_class = (
    classification_pipeline
    .predict_proba(X_val_c)[:, 1]
)


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_val_c,
    y_pred_class
)

precision = precision_score(
    y_val_c,
    y_pred_class
)

recall = recall_score(
    y_val_c,
    y_pred_class
)

f1 = f1_score(
    y_val_c,
    y_pred_class
)

roc_auc = roc_auc_score(
    y_val_c,
    y_prob_class
)


print("\n--- Classification Metrics ---")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# REGRESSION
# ============================================================

regression_drop = [
    "StudentID",
    "PlacementStatus",
    "IsAnomaly",
    "Salary Package"
]

regression_features = [
    c for c in df.columns
    if c not in regression_drop
]
print("\nRegression features:")
print(regression_features)

print(
    "Salary Package included:",
    "Salary Package" in regression_features
)

X_reg = df[
    regression_features
].copy()

y_reg = df[
    "Salary Package"
].copy()


# ------------------------------------------------------------
# Regression preprocessing
# ------------------------------------------------------------

reg_num_features = (
    X_reg
    .select_dtypes(include=[np.number])
    .columns
    .tolist()
)

reg_cat_features = (
    X_reg
    .select_dtypes(exclude=[np.number])
    .columns
    .tolist()
)


reg_num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


reg_cat_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


reg_preprocessor = ColumnTransformer([
    (
        "num",
        reg_num_pipeline,
        reg_num_features
    ),
    (
        "cat",
        reg_cat_pipeline,
        reg_cat_features
    )
])


regression_pipeline = Pipeline([
    (
        "preprocessor",
        reg_preprocessor
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
])


# ------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------

X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# Train regression model
# ------------------------------------------------------------

regression_pipeline.fit(
    X_train_r,
    y_train_r
)


y_pred_reg = (
    regression_pipeline
    .predict(X_val_r)
)


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_val_r,
    y_pred_reg
)

mse = mean_squared_error(
    y_val_r,
    y_pred_reg
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_val_r,
    y_pred_reg
)


print("\n--- Regression Metrics ---")

print(
    f"MAE : {mae:.4f}"
)

print(
    f"MSE : {mse:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R2  : {r2:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "session": "M5-3",

    "classification": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    },

    "regression": {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }
}


results_path = (
    MODEL_DIR /
    "m5_3_results.json"
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC"
]

metric_values = [
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
]

bars = axes[0].bar(
    metric_names,
    metric_values
)

axes[0].set_title(
    "M5-3: Classification Metrics"
)

axes[0].set_ylabel(
    "Score"
)

axes[0].set_ylim(
    0,
    1.05
)

for bar, value in zip(
    bars,
    metric_values
):
    axes[0].text(
        bar.get_x() +
        bar.get_width() / 2,
        value + 0.01,
        f"{value:.4f}",
        ha="center"
    )


# ------------------------------------------------------------
# Regression actual vs predicted
# ------------------------------------------------------------

axes[1].scatter(
    y_val_r,
    y_pred_reg,
    alpha=0.6
)

max_value = max(
    y_val_r.max(),
    y_pred_reg.max()
)

axes[1].plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

axes[1].set_title(
    f"M5-3: Salary Prediction "
    f"(RMSE={rmse:.2f}, R²={r2:.4f})"
)

axes[1].set_xlabel(
    "Actual Salary Package (LPA)"
)

axes[1].set_ylabel(
    "Predicted Salary Package (LPA)"
)


plt.tight_layout()


plot_path = (
    PLOT_DIR /
    "M5_3_Classification_and_Regression_Metrics.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")