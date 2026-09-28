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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


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

print("=== M5-6: Probability Calibration + Platt Scaling ===")
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
    include=["object"]
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


# ============================================================
# RAW PROBABILITY METRICS
# ============================================================

brier_raw = brier_score_loss(
    y_val,
    prob_raw
)

logloss_raw = log_loss(
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


# ============================================================
# CALIBRATED METRICS
# ============================================================

brier_platt = brier_score_loss(
    y_val,
    prob_platt
)

logloss_platt = log_loss(
    y_val,
    prob_platt
)

error_reduction = (
    (brier_raw - brier_platt)
    / brier_raw
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n--- Probability Calibration ---")

print(
    f"Raw RF -> Brier Score: {brier_raw:.4f} "
    f"| Log Loss: {logloss_raw:.4f}"
)

print(
    f"Platt Scaled -> Brier Score: {brier_platt:.4f} "
    f"({error_reduction:.2f}% Brier error reduction)"
)

print(
    f"Platt Scaled -> Log Loss: {logloss_platt:.4f}"
)


# ============================================================
# RELIABILITY DIAGRAM
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


plt.figure(figsize=(7, 6))

plt.plot(
    [0, 1],
    [0, 1],
    "k:",
    label="Perfectly Calibrated"
)

plt.plot(
    mean_prob_raw,
    fraction_raw,
    "s-",
    label=f"Raw RF (Brier = {brier_raw:.4f})"
)

plt.plot(
    mean_prob_platt,
    fraction_platt,
    "o-",
    label=f"Platt Scaled (Brier = {brier_platt:.4f})"
)

plt.title(
    "M5-6: Reliability Diagram — Platt Scaling"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Actual Positives")

plt.legend(loc="upper left")
plt.tight_layout()


plot_path = os.path.join(
    PLOT_DIR,
    "M5_6_Probability_Calibration_Platt.png"
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
    "log_loss_raw": float(logloss_raw),
    "brier_platt": float(brier_platt),
    "log_loss_platt": float(logloss_platt),
    "brier_error_reduction_percent": float(error_reduction)
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_6_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")