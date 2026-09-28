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
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    average_precision_score
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

print("=== M5-5: Precision-Recall + Class Imbalance ===")
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
# PROBABILITIES
# ============================================================

y_prob = (
    pipeline
    .predict_proba(X_val)[:, 1]
)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds = (
    precision_recall_curve(
        y_val,
        y_prob
    )
)


# ============================================================
# PR-AUC + AP
# ============================================================

pr_auc = auc(
    recall,
    precision
)

average_precision = (
    average_precision_score(
        y_val,
        y_prob
    )
)


# ============================================================
# CLASS PREVALENCE
# ============================================================

prevalence = np.mean(y_val)


# ============================================================
# F1 ACROSS THRESHOLDS
# ============================================================

f1_scores = (
    2 *
    (
        precision[:-1] *
        recall[:-1]
    )
    /
    (
        precision[:-1] +
        recall[:-1] +
        1e-10
    )
)


peak_f1_index = np.argmax(
    f1_scores
)

optimal_threshold = (
    thresholds[peak_f1_index]
)


optimal_precision = (
    precision[peak_f1_index]
)

optimal_recall = (
    recall[peak_f1_index]
)

optimal_f1 = (
    f1_scores[peak_f1_index]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n--- Precision-Recall Analysis ---")

print(
    f"Prevalence (pi): "
    f"{prevalence:.4f}"
)

print(
    f"PR-AUC: "
    f"{pr_auc:.4f}"
)

print(
    f"AP Score: "
    f"{average_precision:.4f}"
)

print(
    f"Optimal PR Decision Cutoff "
    f"(Peak F1): "
    f"{optimal_threshold:.4f}"
)

print(
    f"Precision at Peak F1: "
    f"{optimal_precision:.4f}"
)

print(
    f"Recall at Peak F1: "
    f"{optimal_recall:.4f}"
)

print(
    f"Peak F1: "
    f"{optimal_f1:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "session": "M5-5",

    "prevalence": float(
        prevalence
    ),

    "pr_auc": float(
        pr_auc
    ),

    "average_precision": float(
        average_precision
    ),

    "optimal_threshold": float(
        optimal_threshold
    ),

    "optimal_precision": float(
        optimal_precision
    ),

    "optimal_recall": float(
        optimal_recall
    ),

    "optimal_f1": float(
        optimal_f1
    )
}


results_path = (
    MODEL_DIR /
    "m5_5_results.json"
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
# PR CURVE
# ------------------------------------------------------------

axes[0].plot(
    recall,
    precision,
    linewidth=2.5,
    label=f"PR Curve (AUC = {pr_auc:.4f})"
)

axes[0].axhline(
    prevalence,
    linestyle="--",
    label=(
        f"Prevalence Baseline "
        f"(pi = {prevalence:.2f})"
    )
)

axes[0].scatter(
    optimal_recall,
    optimal_precision,
    s=120,
    zorder=5,
    label=(
        f"Peak F1 Cutoff "
        f"(tau = {optimal_threshold:.2f})"
    )
)

axes[0].set_title(
    "M5-5: Precision-Recall Curve"
)

axes[0].set_xlabel(
    "Recall"
)

axes[0].set_ylabel(
    "Precision"
)

axes[0].legend(
    loc="lower left"
)


# ------------------------------------------------------------
# METRIC TRADE-OFFS
# ------------------------------------------------------------

axes[1].plot(
    thresholds,
    precision[:-1],
    label="Precision",
    linewidth=2
)

axes[1].plot(
    thresholds,
    recall[:-1],
    label="Recall",
    linewidth=2
)

axes[1].plot(
    thresholds,
    f1_scores,
    label="F1-Score",
    linestyle=":",
    linewidth=2
)

axes[1].axvline(
    optimal_threshold,
    linestyle="--",
    label=(
        f"Optimal Cutoff "
        f"tau = {optimal_threshold:.2f}"
    )
)

axes[1].set_title(
    "M5-5: Metric Trade-offs"
)

axes[1].set_xlabel(
    "Decision Cutoff Threshold"
)

axes[1].set_ylabel(
    "Score"
)

axes[1].legend(
    loc="lower left"
)


plt.tight_layout()


plot_path = (
    PLOT_DIR /
    "M5_5_Precision_Recall_Curve.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")