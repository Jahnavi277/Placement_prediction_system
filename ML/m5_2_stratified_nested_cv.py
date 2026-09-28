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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "placement_predict_cleaned.csv"
PLOT_DIR = ROOT / "ML" / "plots"
MODEL_DIR = ROOT / "ML" / "models"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

target = "PlacementStatus"

drop_cols = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
    "PlacementStatus",
    "Salary Package",
]

feature_cols = [
    c for c in df.columns
    if c not in drop_cols
]

X = df[feature_cols].copy()
y = df[target].copy()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()


num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])


cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )),
])


preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features),
])


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )),
])


# ============================================================
# DEVELOPMENT / TEST SPLIT
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# M5-2A: STRATIFIED 5-FOLD CV
# ============================================================

stratified_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

stratified_scores = cross_val_score(
    pipeline,
    X_dev,
    y_dev,
    cv=stratified_cv,
    scoring="accuracy"
)

print("=== M5-2: Stratified CV + Nested CV ===")

print(
    "5-Fold Stratified CV Accuracy:",
    np.round(stratified_scores, 4)
)

print(
    f"Mean Stratified Accuracy: "
    f"{stratified_scores.mean():.4f}"
)

print(
    f"2 x Std Dev: "
    f"{2 * stratified_scores.std():.4f}"
)


# ============================================================
# CHECK CLASS RATIO PRESERVATION
# ============================================================

stratified_ratios = []

for _, val_idx in stratified_cv.split(X_dev, y_dev):
    fold_y = y_dev.iloc[val_idx]
    stratified_ratios.append(
        fold_y.mean()
    )

overall_ratio = y_dev.mean()

print(
    f"Overall positive-class ratio: "
    f"{overall_ratio:.4f}"
)

print(
    "Fold positive-class ratios:",
    np.round(stratified_ratios, 4)
)


# ============================================================
# M5-2B: NESTED CROSS-VALIDATION
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE
)

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Faculty hyperparameter grid.
param_grid = {
    "model__C": [
        0.01,
        0.1,
        1.0,
        10.0
    ]
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="f1",
    n_jobs=-1
)


nested_scores = cross_val_score(
    grid_search,
    X,
    y,
    cv=outer_cv,
    scoring="f1",
    n_jobs=-1
)


print(
    "Nested CV Outer F1 Scores:",
    np.round(nested_scores, 4)
)

print(
    f"Nested CV Mean F1: "
    f"{nested_scores.mean():.4f}"
)

print(
    f"Nested CV 2 x Std Dev: "
    f"{2 * nested_scores.std():.4f}"
)


# ============================================================
# FINAL GRID SEARCH ON DEVELOPMENT SET
# ============================================================

final_grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="f1",
    n_jobs=-1
)

final_grid.fit(
    X_dev,
    y_dev
)

print(
    f"Best C from development data: "
    f"{final_grid.best_params_['model__C']}"
)

print(
    f"Best inner-CV F1: "
    f"{final_grid.best_score_:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "session": "M5-2",
    "random_state": RANDOM_STATE,

    "stratified_cv": {
        "folds": 5,
        "scores": [
            float(x)
            for x in stratified_scores
        ],
        "mean_accuracy": float(
            stratified_scores.mean()
        ),
        "std_accuracy": float(
            stratified_scores.std()
        ),
    },

    "class_ratio": {
        "overall": float(overall_ratio),
        "fold_ratios": [
            float(x)
            for x in stratified_ratios
        ],
    },

    "nested_cv": {
        "inner_folds": 3,
        "outer_folds": 5,
        "parameter_grid": {
            "C": [
                0.01,
                0.1,
                1.0,
                10.0
            ]
        },
        "outer_f1_scores": [
            float(x)
            for x in nested_scores
        ],
        "mean_f1": float(
            nested_scores.mean()
        ),
        "std_f1": float(
            nested_scores.std()
        ),
    },

    "final_development_search": {
        "best_C": float(
            final_grid.best_params_["model__C"]
        ),
        "best_inner_cv_f1": float(
            final_grid.best_score_
        ),
    },
}


results_path = MODEL_DIR / "m5_2_results.json"

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
    figsize=(12, 5)
)


# Class ratio plot
fold_numbers = np.arange(1, 6)

axes[0].bar(
    fold_numbers,
    stratified_ratios
)

axes[0].axhline(
    overall_ratio,
    linestyle="--",
    label=f"Overall: {overall_ratio:.3f}"
)

axes[0].set_title(
    "M5-2: Stratified Class Ratio Preservation"
)

axes[0].set_xlabel(
    "Fold"
)

axes[0].set_ylabel(
    "Positive Class Fraction"
)

axes[0].set_xticks(
    fold_numbers
)

axes[0].legend()


# Nested CV plot
axes[1].bar(
    fold_numbers,
    nested_scores
)

axes[1].axhline(
    nested_scores.mean(),
    linestyle="--",
    label=f"Mean F1: {nested_scores.mean():.4f}"
)

axes[1].set_title(
    "M5-2: Nested CV Outer F1"
)

axes[1].set_xlabel(
    "Outer Fold"
)

axes[1].set_ylabel(
    "F1 Score"
)

axes[1].set_xticks(
    fold_numbers
)

axes[1].legend()


plt.tight_layout()

plot_path = (
    PLOT_DIR /
    "M5_2_Stratified_and_Nested_CV.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"Saved: {results_path}")
print(f"Saved: {plot_path}")