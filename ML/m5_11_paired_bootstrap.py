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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score


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

print("=== M5-11: Paired Bootstrap Comparison ===")
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
# Same structure as M5-10
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
# ============================================================

model_a = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ============================================================
# MODEL B — GRADIENT BOOSTING
# ============================================================

model_b = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingClassifier(
        random_state=42
    ))
])


# ============================================================
# TRAIN BOTH MODELS
# ============================================================

model_a.fit(
    X_train,
    y_train
)

model_b.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

pred_a = model_a.predict(X_val)
pred_b = model_b.predict(X_val)

y_val_array = y_val.to_numpy()


# ============================================================
# ORIGINAL F1 SCORES
# ============================================================

original_f1_a = f1_score(
    y_val_array,
    pred_a
)

original_f1_b = f1_score(
    y_val_array,
    pred_b
)

print("\n--- Original Validation F1 ---")
print(f"Model A F1: {original_f1_a:.4f}")
print(f"Model B F1: {original_f1_b:.4f}")
print(
    f"Original F1 Difference (A - B): "
    f"{original_f1_a - original_f1_b:.4f}"
)


# ============================================================
# PAIRED BOOTSTRAP
# Faculty: B = 1000
# ============================================================

B_bootstrap = 1000

np.random.seed(42)

n_val = len(y_val_array)

boot_diffs = []


for _ in range(B_bootstrap):

    # Resample the SAME indices for both models
    boot_idx = np.random.choice(
        n_val,
        size=n_val,
        replace=True
    )

    f1_a = f1_score(
        y_val_array[boot_idx],
        pred_a[boot_idx]
    )

    f1_b = f1_score(
        y_val_array[boot_idx],
        pred_b[boot_idx]
    )

    boot_diffs.append(
        f1_a - f1_b
    )


boot_diffs = np.array(
    boot_diffs
)


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

mean_difference = np.mean(
    boot_diffs
)

ci_lower = np.percentile(
    boot_diffs,
    2.5
)

ci_upper = np.percentile(
    boot_diffs,
    97.5
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    f"\nPaired Bootstrap Resampling "
    f"(B={B_bootstrap}):"
)

print(
    f"Mean F1 Difference (A - B): "
    f"{mean_difference:.4f}"
)

print(
    f"95% Non-Parametric CI: "
    f"[{ci_lower:.4f}, {ci_upper:.4f}]"
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    boot_diffs,
    bins=30,
    edgecolor="black"
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Difference"
)

ax.axvline(
    ci_lower,
    linestyle=":",
    linewidth=2,
    label=f"95% CI Lower: {ci_lower:.4f}"
)

ax.axvline(
    ci_upper,
    linestyle=":",
    linewidth=2,
    label=f"95% CI Upper: {ci_upper:.4f}"
)

ax.set_title(
    f"M5-11: Paired Bootstrap F1 Differences "
    f"(B={B_bootstrap})"
)

ax.set_xlabel(
    "F1 Difference (Model A - Model B)"
)

ax.set_ylabel(
    "Resampling Frequency"
)

ax.legend(
    loc="upper right"
)

plt.tight_layout()


# ============================================================
# SAVE PLOT
# ============================================================

plot_path = os.path.join(
    PLOT_DIR,
    "M5_11_Paired_Bootstrap_Distribution.png"
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
    "bootstrap_iterations": B_bootstrap,

    "model_a": "LogisticRegression",
    "model_b": "GradientBoostingClassifier",

    "original_validation_f1": {
        "model_a": float(original_f1_a),
        "model_b": float(original_f1_b),
        "difference_a_minus_b": float(
            original_f1_a - original_f1_b
        )
    },

    "paired_bootstrap": {
        "mean_f1_difference_a_minus_b": float(
            mean_difference
        ),
        "ci_95_lower": float(
            ci_lower
        ),
        "ci_95_upper": float(
            ci_upper
        )
    }
}

results_path = os.path.join(
    MODEL_DIR,
    "m5_11_results.json"
)

with open(
    results_path,
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )


print(f"\nSaved: {results_path}")
print(f"Saved: {plot_path}")

print("\n" + "=" * 70)
print("M5-11 COMPLETED")
print("=" * 70)