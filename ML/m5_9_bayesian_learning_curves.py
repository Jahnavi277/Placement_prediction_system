import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from scipy.stats import norm


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

print("=== M5-9: Bayesian Optimization + Learning Curves ===")
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
# CLASSIFICATION PIPELINE
# Faculty M5 pipeline uses Logistic Regression here
# ============================================================

pipeline_m51 = Pipeline([
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
# M5-9 BAYESIAN OPTIMIZATION
# Faculty demonstration data
# ============================================================

X_bo = np.array([
    [3],
    [5],
    [8],
    [12],
    [15],
    [20]
])

y_bo = np.array([
    0.880,
    0.925,
    0.938,
    0.945,
    0.941,
    0.939
])


# Gaussian Process surrogate

gp_kernel = (
    ConstantKernel(
        1.0,
        (1e-3, 1e3)
    )
    *
    RBF(
        length_scale=5.0,
        length_scale_bounds=(1e-2, 1e2)
    )
)

gp_model = GaussianProcessRegressor(
    kernel=gp_kernel,
    alpha=1e-4,
    n_restarts_optimizer=10,
    random_state=42
)

gp_model.fit(
    X_bo,
    y_bo
)


# ============================================================
# PREDICTION GRID
# ============================================================

X_grid = np.linspace(
    1,
    25,
    200
).reshape(-1, 1)

y_pred_gp, sigma_gp = gp_model.predict(
    X_grid,
    return_std=True
)


# ============================================================
# EXPECTED IMPROVEMENT
# ============================================================

best_y = np.max(y_bo)

xi = 0.01

improvement = (
    y_pred_gp
    - best_y
    - xi
)

Z = improvement / (
    sigma_gp + 1e-9
)

ei = (
    improvement * norm.cdf(Z)
    + sigma_gp * norm.pdf(Z)
)

opt_bo_depth = X_grid[
    np.argmax(ei)
][0]

print(
    f"Bayesian GP Optimization Suggested "
    f"Max Depth: {opt_bo_depth:.1f}"
)


# ============================================================
# LEARNING CURVES
# ============================================================

train_sizes, train_scores, val_scores = learning_curve(
    pipeline_m51,
    X_dev,
    y_dev,
    cv=stratified_cv,
    train_sizes=np.linspace(
        0.1,
        1.0,
        5
    ),
    scoring="accuracy",
    n_jobs=-1
)


# Mean and standard deviation

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
# PRINT LEARNING CURVE RESULTS
# ============================================================

print("\n--- Learning Curve Results ---")

for size, train_score, val_score in zip(
    train_sizes,
    train_mean,
    val_mean
):
    print(
        f"Training samples: {size:,.0f} | "
        f"Training Accuracy: {train_score:.4f} | "
        f"CV Accuracy: {val_score:.4f}"
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
# Bayesian Optimization
# ------------------------------------------------------------

axes[0].plot(
    X_grid,
    y_pred_gp,
    label="GP Mean Surrogate"
)

axes[0].fill_between(
    X_grid.ravel(),
    y_pred_gp - 1.96 * sigma_gp,
    y_pred_gp + 1.96 * sigma_gp,
    alpha=0.2,
    label="95% Confidence"
)

axes[0].scatter(
    X_bo,
    y_bo,
    s=50,
    zorder=5,
    label="Sampled Evaluated Depths"
)

axes[0].plot(
    X_grid,
    best_y + ei * 5,
    linestyle="--",
    label="EI Acquisition (Scaled)"
)

axes[0].axvline(
    opt_bo_depth,
    linestyle=":",
    label=f"Next Sample Point ({opt_bo_depth:.1f})"
)

axes[0].set_title(
    "M5-9: Bayesian Optimization via Gaussian Process"
)

axes[0].set_xlabel(
    "Hyperparameter: Max Depth"
)

axes[0].set_ylabel(
    "F1-Score / Acquisition"
)

axes[0].legend(
    loc="lower right"
)


# ------------------------------------------------------------
# Learning Curves
# ------------------------------------------------------------

axes[1].plot(
    train_sizes,
    train_mean,
    "o-",
    label="Training Score"
)

axes[1].fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15
)

axes[1].plot(
    train_sizes,
    val_mean,
    "o-",
    label="Cross-Validation Score"
)

axes[1].fill_between(
    train_sizes,
    val_mean - val_std,
    val_mean + val_std,
    alpha=0.15
)

axes[1].set_title(
    "M5-9: Learning Curves (Sample Size N vs Accuracy)"
)

axes[1].set_xlabel(
    "Training Sample Size (N)"
)

axes[1].set_ylabel(
    "Accuracy Score"
)

axes[1].legend(
    loc="lower right"
)


plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plot_path = os.path.join(
    PLOT_DIR,
    "M5_9_Bayesian_Opt_and_Learning_Curves.png"
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
    "bayesian_optimization": {
        "faculty_demo_depths": [
            3, 5, 8, 12, 15, 20
        ],
        "faculty_demo_f1": [
            0.880,
            0.925,
            0.938,
            0.945,
            0.941,
            0.939
        ],
        "suggested_next_max_depth": float(
            opt_bo_depth
        )
    },
    "learning_curve": {
        "training_sizes": [
            int(x) for x in train_sizes
        ],
        "training_accuracy_mean": [
            float(x) for x in train_mean
        ],
        "training_accuracy_std": [
            float(x) for x in train_std
        ],
        "cv_accuracy_mean": [
            float(x) for x in val_mean
        ],
        "cv_accuracy_std": [
            float(x) for x in val_std
        ]
    }
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_9_results.json"
)

with open(results_path, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")