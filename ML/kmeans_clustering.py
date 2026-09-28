import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE_SIZE = 5000

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_cleaned.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "ML", "models")
PLOT_DIR = os.path.join(BASE_DIR, "ML", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. SELECT NUMERICAL FEATURES
# ============================================================

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Do not allow ID or prediction targets to influence clustering.
EXCLUDE_COLUMNS = [
    "StudentID",
    "PlacementStatus",
    "Salary Package"
]

feature_cols = [
    col for col in numeric_cols
    if col not in EXCLUDE_COLUMNS
]

print("\nFeatures used for clustering:")

for col in feature_cols:
    print(f"  - {col}")

X_raw = df[feature_cols].copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before imputation:")

missing = X_raw.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("  None")
else:
    print(missing)

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X_raw),
    columns=feature_cols,
    index=X_raw.index
)

print(
    "\nMissing values after imputation:",
    X_imputed.isna().sum().sum()
)


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

print(
    f"\nScaled data shape: {X_scaled.shape}"
)

print(
    f"Mean: {X_scaled.mean():.4f}"
)

print(
    f"Std:  {X_scaled.std():.4f}"
)


# ============================================================
# 5. K-MEANS FOR K = 2 ... 10
# ============================================================

results = []

rng = np.random.RandomState(RANDOM_STATE)

sample_size = min(
    SILHOUETTE_SAMPLE_SIZE,
    len(X_scaled)
)

sample_idx = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

print("\n" + "=" * 60)
print("K SELECTION")
print("=" * 60)

for k in K_RANGE:

    print(f"\nRunning K = {k}...")

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_

    silhouette = silhouette_score(
        X_scaled[sample_idx],
        labels[sample_idx]
    )

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette": silhouette
    })

    print(
        f"Inertia = {inertia:,.2f} | "
        f"Silhouette = {silhouette:.4f}"
    )


results_df = pd.DataFrame(results)


# ============================================================
# 6. SELECT OPTIMAL K
# ============================================================

best_k_silhouette = int(
    results_df.loc[
        results_df["Silhouette"].idxmax(),
        "K"
    ]
)

# Numerical elbow calculation
inertia_values = results_df["Inertia"].values

first_diff = np.diff(inertia_values)

second_diff = np.diff(first_diff)

elbow_k = int(
    results_df["K"].values[
        np.argmax(second_diff) + 2
    ]
)

OPTIMAL_K = best_k_silhouette

print("\n" + "=" * 60)
print("OPTIMAL K")
print("=" * 60)

print(
    f"Best K by silhouette : {best_k_silhouette}"
)

print(
    f"Best silhouette      : "
    f"{results_df['Silhouette'].max():.4f}"
)

print(
    f"Elbow K              : {elbow_k}"
)

print(
    f"\nFINAL K = {OPTIMAL_K}"
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "kmeans_k_selection_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

optimal_k_path = os.path.join(
    MODEL_DIR,
    "optimal_k.json"
)

with open(optimal_k_path, "w") as f:

    json.dump(
        {
            "optimal_k": OPTIMAL_K,
            "silhouette_score": float(
                results_df["Silhouette"].max()
            ),
            "elbow_k": elbow_k,
            "random_state": RANDOM_STATE,
            "features": feature_cols
        },
        f,
        indent=4
    )


# ============================================================
# 8. ELBOW + SILHOUETTE PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


# Elbow
axes[0].plot(
    results_df["K"],
    results_df["Inertia"],
    marker="o"
)

axes[0].set_title(
    "Elbow Curve — Inertia vs K"
)

axes[0].set_xlabel(
    "Number of clusters (K)"
)

axes[0].set_ylabel(
    "Inertia"
)

axes[0].set_xticks(
    list(K_RANGE)
)

axes[0].grid(
    alpha=0.3
)


# Silhouette
axes[1].plot(
    results_df["K"],
    results_df["Silhouette"],
    marker="o"
)

axes[1].axvline(
    OPTIMAL_K,
    linestyle="--",
    label=f"Optimal K = {OPTIMAL_K}"
)

axes[1].set_title(
    "Silhouette Score vs K"
)

axes[1].set_xlabel(
    "Number of clusters (K)"
)

axes[1].set_ylabel(
    "Mean silhouette score"
)

axes[1].set_xticks(
    list(K_RANGE)
)

axes[1].grid(
    alpha=0.3
)

axes[1].legend()


plt.tight_layout()

plot_path = os.path.join(
    PLOT_DIR,
    "elbow_and_silhouette_curves.png"
)

plt.savefig(
    plot_path,
    dpi=150
)

plt.close()


# ============================================================
# 9. FINAL K-MEANS MODEL
# ============================================================

final_model = KMeans(
    n_clusters=OPTIMAL_K,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)

df["Cluster"] = final_model.fit_predict(
    X_scaled
)


# ============================================================
# 10. SAVE CLUSTERED DATA
# ============================================================

clustered_path = os.path.join(
    "Data",
    "placement_data_with_clusters.csv"
)

clustered_path = os.path.join(
    BASE_DIR,
    clustered_path
)

df.to_csv(
    clustered_path,
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("K-MEANS COMPLETE")
print("=" * 60)

print(f"Optimal K       : {OPTIMAL_K}")
print(f"Results saved   : {results_path}")
print(f"Metadata saved  : {optimal_k_path}")
print(f"Plot saved      : {plot_path}")
print(f"Clustered data  : {clustered_path}")