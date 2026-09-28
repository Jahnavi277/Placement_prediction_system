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
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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
# M5-1: TRAIN / VALIDATION / TEST + 5-FOLD CV
# ============================================================

df = pd.read_csv(DATA_PATH)

target = "PlacementStatus"

# Same exclusions used by the faculty M5 master pipeline.
drop_cols = [
    "StudentID",
    "CGPA_Tier",
    "IsAnomaly",
    "PlacementStatus",
    "Salary Package"
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df[target].copy()


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])


# ------------------------------------------------------------
# STEP 1: LOCK TEST SET
# ------------------------------------------------------------

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ------------------------------------------------------------
# STEP 2: TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_dev
)


print("=== M5-1: Train / Validation / Test + 5-Fold CV ===")

print(f"Full dataset : {len(X)}")
print(f"Train        : {len(X_train)} (60%)")
print(f"Validation   : {len(X_val)} (20%)")
print(f"Test         : {len(X_test)} (20%)")
print(f"Features     : {len(feature_cols)}")


# ------------------------------------------------------------
# STEP 3: 5-FOLD CROSS VALIDATION
# ------------------------------------------------------------

cv_5fold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    pipeline,
    X_dev,
    y_dev,
    cv=cv_5fold,
    scoring="accuracy"
)

print(
    "5-Fold CV Accuracy:",
    np.round(cv_scores, 4)
)

print(
    f"Mean CV Accuracy   : {cv_scores.mean():.4f}"
)

print(
    f"2 x Std Dev        : {2 * cv_scores.std():.4f}"
)


# ------------------------------------------------------------
# STEP 4: FINAL TEST EVALUATION
# ------------------------------------------------------------

pipeline.fit(X_dev, y_dev)

test_accuracy = pipeline.score(
    X_test,
    y_test
)

print(
    f"Vault-Locked Test Accuracy: "
    f"{test_accuracy:.4f}"
)


# ------------------------------------------------------------
# SAVE SPLIT INDICES
# ------------------------------------------------------------

with open(
    MODEL_DIR / "m5_1_split_indices.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "train_indices": X_train.index.tolist(),
            "validation_indices": X_val.index.tolist(),
            "test_indices": X_test.index.tolist()
        },
        f
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

with open(
    MODEL_DIR / "m5_1_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "session": "M5-1",
            "random_state": RANDOM_STATE,
            "dataset_rows": int(len(df)),
            "feature_count": int(len(feature_cols)),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "cv_type": "KFold",
            "cv_folds": 5,
            "cv_scores": [
                float(v)
                for v in cv_scores
            ],
            "cv_mean_accuracy": float(
                cv_scores.mean()
            ),
            "cv_std_accuracy": float(
                cv_scores.std()
            ),
            "test_accuracy": float(
                test_accuracy
            )
        },
        f,
        indent=2
    )


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

sizes = [
    len(X_train),
    len(X_val),
    len(X_test)
]

labels = [
    "Train (60%)",
    "Validation (20%)",
    "Test (20%)"
]

axes[0].bar(
    labels,
    sizes
)

axes[0].set_title(
    "M5-1: Three-Way Data Partition"
)

axes[0].set_ylabel(
    "Student Records"
)

for i, value in enumerate(sizes):
    axes[0].text(
        i,
        value,
        str(value),
        ha="center",
        va="bottom"
    )


axes[1].plot(
    range(1, 6),
    cv_scores,
    marker="o",
    label="Fold Accuracy"
)

axes[1].axhline(
    cv_scores.mean(),
    linestyle="--",
    label=f"Mean CV: {cv_scores.mean():.4f}"
)

axes[1].set_title(
    "M5-1: 5-Fold CV Accuracy"
)

axes[1].set_xlabel("Fold")

axes[1].set_ylabel("Accuracy")

axes[1].set_xticks(
    range(1, 6)
)

axes[1].legend()

plt.tight_layout()

plot_path = (
    PLOT_DIR /
    "M5_1_Data_Split_and_CV.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print(f"Saved: {plot_path}")

print(
    f"Saved: "
    f"{MODEL_DIR / 'm5_1_results.json'}"
)

print(
    f"Saved: "
    f"{MODEL_DIR / 'm5_1_split_indices.json'}"
)