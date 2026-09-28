import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    validation_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


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

print("=== M5-10: Validation Curves + McNemar's Test ===")
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
# DEVELOPMENT / TEST SPLIT
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# Used for McNemar's test
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
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
# MODEL A — LOGISTIC REGRESSION
# Faculty pipeline_m51
# ============================================================

model_a = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ============================================================
# STRATIFIED CV
# ============================================================

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# M5-10 PART 1
# VALIDATION CURVE
# ============================================================

param_range = np.logspace(
    -3,
    2,
    6
)

train_scores, val_scores = validation_curve(
    model_a,
    X_dev,
    y_dev,
    param_name="model__C",
    param_range=param_range,
    cv=stratified_cv,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = np.mean(
    train_scores,
    axis=1
)

train_std = np.std(
    train_scores,
    axis=1
)

val_mean = np.mean(
    val_scores,
    axis=1
)

val_std = np.std(
    val_scores,
    axis=1
)


# ============================================================
# PRINT VALIDATION CURVE
# ============================================================

print("\n--- Validation Curve ---")

for c, train_score, validation_score in zip(
    param_range,
    train_mean,
    val_mean
):
    print(
        f"C = {c:.4f} | "
        f"Training Accuracy = {train_score:.4f} | "
        f"Validation Accuracy = {validation_score:.4f}"
    )


# ============================================================
# M5-10 PART 2
# MCNEMAR'S TEST
# ============================================================

print("\n--- McNemar's Test ---")


# Model B — Gradient Boosting
model_b = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])


# Train both models on the SAME training set

model_a.fit(
    X_train,
    y_train
)

model_b.fit(
    X_train,
    y_train
)


# Predictions on the SAME validation set

pred_a = model_a.predict(X_val)
pred_b = model_b.predict(X_val)


# ============================================================
# DISCORDANT PAIRS
# ============================================================

b_mcn = np.sum(
    (pred_a == y_val.to_numpy()) &
    (pred_b != y_val.to_numpy())
)

c_mcn = np.sum(
    (pred_a != y_val.to_numpy()) &
    (pred_b == y_val.to_numpy())
)


# ============================================================
# MCCNEMAR CHI-SQUARE STATISTIC
# Faculty's continuity-corrected formula
# ============================================================

mcn_stat = (
    (abs(b_mcn - c_mcn) - 1) ** 2
    / (b_mcn + c_mcn + 1e-10)
)

p_value = chi2.sf(
    mcn_stat,
    df=1
)


# ============================================================
# PRINT MCNEMAR RESULTS
# ============================================================

print(
    f"McNemar Discordant Pairs -> "
    f"b (A right, B wrong): {b_mcn} | "
    f"c (A wrong, B right): {c_mcn}"
)

print(
    f"McNemar Chi-Square Stat: "
    f"{mcn_stat:.4f}"
)

print(
    f"p-value: {p_value:.4f}"
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
# Validation Curve
# ------------------------------------------------------------

axes[0].semilogx(
    param_range,
    train_mean,
    "o-",
    label="Training Score"
)

axes[0].semilogx(
    param_range,
    val_mean,
    "o-",
    label="Validation Score"
)

axes[0].set_title(
    "M5-10: Validation Curve"
)

axes[0].set_xlabel(
    "Hyperparameter C (Inverse Regularization)"
)

axes[0].set_ylabel(
    "Accuracy Score"
)

axes[0].legend(
    loc="lower right"
)


# ------------------------------------------------------------
# McNemar Discordant Counts
# ------------------------------------------------------------

axes[1].bar(
    [
        "b: A Right, B Wrong",
        "c: A Wrong, B Right"
    ],
    [
        b_mcn,
        c_mcn
    ]
)

axes[1].set_title(
    f"M5-10: McNemar Test "
    f"(p = {p_value:.4f})"
)

axes[1].set_ylabel(
    "Count of Discordant Predictions"
)

for i, value in enumerate([
    b_mcn,
    c_mcn
]):
    axes[1].text(
        i,
        value + 0.2,
        str(value),
        ha="center"
    )


plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plot_path = os.path.join(
    PLOT_DIR,
    "M5_10_Validation_Curves_and_McNemar.png"
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
    "validation_curve": {
        "C_values": [
            float(x) for x in param_range
        ],
        "training_accuracy_mean": [
            float(x) for x in train_mean
        ],
        "training_accuracy_std": [
            float(x) for x in train_std
        ],
        "validation_accuracy_mean": [
            float(x) for x in val_mean
        ],
        "validation_accuracy_std": [
            float(x) for x in val_std
        ]
    },
    "mcnemar_test": {
        "model_a": "LogisticRegression",
        "model_b": "GradientBoostingClassifier",
        "b_discordant": int(b_mcn),
        "c_discordant": int(c_mcn),
        "chi_square": float(mcn_stat),
        "p_value": float(p_value)
    }
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_10_results.json"
)

with open(results_path, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")