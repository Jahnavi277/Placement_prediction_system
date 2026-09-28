import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
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

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=== M5-4: Confusion Matrix + ROC-AUC Deep Dive ===")
print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES
# ============================================================

drop_cols = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
    "PlacementStatus",
    "Salary Package"
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

X = df[feature_cols].copy()
y = df["PlacementStatus"].copy()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = (
    X.select_dtypes(include=[np.number])
    .columns
    .tolist()
)

categorical_features = (
    X.select_dtypes(exclude=[np.number])
    .columns
    .tolist()
)


num_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


cat_pipeline = Pipeline([
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


preprocessor = ColumnTransformer([
    (
        "num",
        num_pipeline,
        numeric_features
    ),
    (
        "cat",
        cat_pipeline,
        categorical_features
    )
])


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_val)

y_prob = (
    pipeline
    .predict_proba(X_val)[:, 1]
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    y_pred
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_val,
    y_prob
)

roc_auc = roc_auc_score(
    y_val,
    y_prob
)


# ============================================================
# YOUDEN'S J
# ============================================================

youden_j = tpr - fpr

optimal_index = np.argmax(
    youden_j
)

optimal_threshold = (
    thresholds[optimal_index]
)


print("\n--- Confusion Matrix ---")
print(cm)

print("\n--- ROC-AUC ---")
print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print("\n--- Optimal Decision Cutoff ---")
print(
    f"Youden's J = TPR - FPR"
)

print(
    f"Optimal threshold: "
    f"{optimal_threshold:.4f}"
)

print(
    f"TPR at threshold: "
    f"{tpr[optimal_index]:.4f}"
)

print(
    f"FPR at threshold: "
    f"{fpr[optimal_index]:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "session": "M5-4",

    "confusion_matrix": cm.tolist(),

    "roc_auc": float(roc_auc),

    "optimal_threshold": float(
        optimal_threshold
    ),

    "optimal_tpr": float(
        tpr[optimal_index]
    ),

    "optimal_fpr": float(
        fpr[optimal_index]
    ),

    "youden_j": float(
        youden_j[optimal_index]
    )
}


results_path = (
    MODEL_DIR /
    "m5_4_results.json"
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
# PLOTS
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[0],
    cbar=False,
    xticklabels=[
        "Not Placed",
        "Placed"
    ],
    yticklabels=[
        "Not Placed",
        "Placed"
    ]
)

axes[0].set_title(
    "M5-4: Confusion Matrix"
)

axes[0].set_xlabel(
    "Predicted Label"
)

axes[0].set_ylabel(
    "True Label"
)


# ------------------------------------------------------------
# ROC Curve
# ------------------------------------------------------------

axes[1].plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"ROC Curve (AUC = {roc_auc:.4f})"
)

axes[1].plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Guess (AUC = 0.50)"
)

axes[1].scatter(
    fpr[optimal_index],
    tpr[optimal_index],
    s=120,
    zorder=5,
    label=(
        f"Optimal Cutoff "
        f"(τ={optimal_threshold:.2f})"
    )
)

axes[1].set_title(
    "M5-4: ROC Curve"
)

axes[1].set_xlabel(
    "False Positive Rate"
)

axes[1].set_ylabel(
    "True Positive Rate"
)

axes[1].legend(
    loc="lower right"
)


plt.tight_layout()


plot_path = (
    PLOT_DIR /
    "M5_4_Confusion_Matrix_and_ROC_Curve.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")