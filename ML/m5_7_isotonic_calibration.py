import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import brier_score_loss


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "Data/placement_predict_cleaned.csv"
MODEL_DIR = "ML/models"
PLOT_DIR = "ML/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=== M5-7: Isotonic Regression + Reliability Diagrams ===")
print(f"Dataset shape: {df.shape}")


# ============================================================
# FEATURES / TARGET
# ============================================================

target = "PlacementStatus"

drop_cols = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
    "PlacementStatus",
    "Salary Package"
]

X = df.drop(columns=drop_cols)
y = df[target]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ============================================================
# RAW RANDOM FOREST
# ============================================================

rf_raw = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_raw.fit(X_train, y_train)

prob_raw = rf_raw.predict_proba(X_val)[:, 1]

brier_raw = brier_score_loss(
    y_val,
    prob_raw
)


# ============================================================
# PLATT SCALING
# ============================================================

calib_platt = CalibratedClassifierCV(
    rf_raw,
    method="sigmoid",
    cv=2
)

calib_platt.fit(X_train, y_train)

prob_platt = calib_platt.predict_proba(X_val)[:, 1]

brier_platt = brier_score_loss(
    y_val,
    prob_platt
)


# ============================================================
# ISOTONIC REGRESSION
# ============================================================

calib_iso = CalibratedClassifierCV(
    rf_raw,
    method="isotonic",
    cv=2
)

calib_iso.fit(X_train, y_train)

prob_iso = calib_iso.predict_proba(X_val)[:, 1]

brier_iso = brier_score_loss(
    y_val,
    prob_iso
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n--- Calibration Results ---")

print(
    f"Uncalibrated RF -> "
    f"Brier Score: {brier_raw:.4f}"
)

print(
    f"Platt Scaling -> "
    f"Brier Score: {brier_platt:.4f}"
)

print(
    f"Isotonic Calibrated -> "
    f"Brier Score: {brier_iso:.4f}"
)


# ============================================================
# RELIABILITY CURVES
# ============================================================

fraction_raw, mean_prob_raw = calibration_curve(
    y_val,
    prob_raw,
    n_bins=10
)

fraction_platt, mean_prob_platt = calibration_curve(
    y_val,
    prob_platt,
    n_bins=10
)

fraction_iso, mean_prob_iso = calibration_curve(
    y_val,
    prob_iso,
    n_bins=10
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# ------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------

axes[0].plot(
    [0, 1],
    [0, 1],
    "k:",
    label="Perfectly Calibrated"
)

axes[0].plot(
    mean_prob_raw,
    fraction_raw,
    "s-",
    label="Uncalibrated RF"
)

axes[0].plot(
    mean_prob_platt,
    fraction_platt,
    "o-",
    label="Platt Scaling (Sigmoid)"
)

axes[0].plot(
    mean_prob_iso,
    fraction_iso,
    "^-",
    label="Isotonic Regression"
)

axes[0].set_title(
    "M5-7: Reliability Diagram Comparison"
)

axes[0].set_xlabel(
    "Mean Predicted Probability"
)

axes[0].set_ylabel(
    "Fraction of Positives"
)

axes[0].legend(loc="upper left")


# ------------------------------------------------------------
# Brier score comparison
# ------------------------------------------------------------

calib_names = [
    "Uncalibrated RF",
    "Platt Scaling",
    "Isotonic Reg"
]

brier_scores = [
    brier_raw,
    brier_platt,
    brier_iso
]

axes[1].bar(
    calib_names,
    brier_scores
)

axes[1].set_title(
    "M5-7: Brier Score Calibration Error Comparison"
)

axes[1].set_ylabel(
    "Brier Score Loss"
)

for i, value in enumerate(brier_scores):
    axes[1].text(
        i,
        value + 0.001,
        f"{value:.4f}",
        ha="center"
    )


plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plot_path = os.path.join(
    PLOT_DIR,
    "M5_7_Isotonic_Regression_Reliability_Diagram.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "dataset_shape": list(df.shape),
    "brier_raw": float(brier_raw),
    "brier_platt": float(brier_platt),
    "brier_isotonic": float(brier_iso)
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_7_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")