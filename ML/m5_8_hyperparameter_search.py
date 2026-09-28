import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier


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

print("=== M5-8: Grid Search + Random Search ===")
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
# RANDOM FOREST PIPELINE
# ============================================================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])


# ============================================================
# INNER CROSS-VALIDATION
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# GRID SEARCH
# ============================================================

param_grid = {
    "model__n_estimators": [20, 50, 100],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="f1",
    n_jobs=-1
)

print("\nRunning GridSearchCV...")

grid_search.fit(
    X_dev,
    y_dev
)


# ============================================================
# RANDOM SEARCH
# ============================================================

param_dist = {
    "model__n_estimators": [20, 50, 100, 150],
    "model__max_depth": [3, 5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=inner_cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

print("Running RandomizedSearchCV...")

random_search.fit(
    X_dev,
    y_dev
)


# ============================================================
# RESULTS
# ============================================================

print("\n--- Hyperparameter Search Results ---")

print(
    "GridSearchCV Best Params :",
    grid_search.best_params_
)

print(
    f"GridSearchCV Best F1      : "
    f"{grid_search.best_score_:.4f}"
)

print(
    "RandomizedSearchCV Best Params :",
    random_search.best_params_
)

print(
    f"RandomizedSearchCV Best F1     : "
    f"{random_search.best_score_:.4f}"
)


# ============================================================
# GRID SEARCH HEATMAP DATA
# ============================================================

results_df = pd.DataFrame(
    grid_search.cv_results_
)

pivot_table = results_df.pivot_table(
    index="param_model__max_depth",
    columns="param_model__n_estimators",
    values="mean_test_score"
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
# Grid vs Random F1
# ------------------------------------------------------------

axes[0].bar(
    [
        "GridSearchCV",
        "RandomizedSearchCV"
    ],
    [
        grid_search.best_score_,
        random_search.best_score_
    ]
)

axes[0].set_title(
    "M5-8: Best F1-Score Comparison"
)

axes[0].set_ylabel(
    "Cross-Validation F1-Score"
)

axes[0].set_ylim(
    0.85,
    1.0
)

for i, value in enumerate([
    grid_search.best_score_,
    random_search.best_score_
]):
    axes[0].text(
        i,
        value + 0.002,
        f"{value:.4f}",
        ha="center"
    )


# ------------------------------------------------------------
# Grid Search Heatmap
# ------------------------------------------------------------

sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".4f",
    cmap="Blues",
    ax=axes[1]
)

axes[1].set_title(
    "M5-8: Grid Search F1-Score Heatmap"
)

axes[1].set_xlabel(
    "Number of Estimators"
)

axes[1].set_ylabel(
    "Max Depth"
)


plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plot_path = os.path.join(
    PLOT_DIR,
    "M5_8_Hyperparameter_Search_Grid_vs_Random.png"
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
    "grid_search": {
        "best_params": grid_search.best_params_,
        "best_f1": float(grid_search.best_score_)
    },
    "random_search": {
        "best_params": random_search.best_params_,
        "best_f1": float(random_search.best_score_)
    }
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_8_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")