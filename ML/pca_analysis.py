import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Data",
    "placement_predict_cleaned.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "models"
)

PLOT_DIR = os.path.join(
    BASE_DIR,
    "ML",
    "plots"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD + PREPROCESS
# ============================================================

print("=" * 65)
print("PCA ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(
    f"\nLoaded dataset: "
    f"{df.shape[0]:,} rows x {df.shape[1]} columns"
)


# Faculty uses numerical features and removes StudentID.
# For PlacementPredict, also exclude target/outcome columns
# so PCA is not influenced by the answers we want to predict.

numeric_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()

EXCLUDE_COLUMNS = [
    "StudentID",
    "PlacementStatus",
    "Salary Package"
]

feature_cols = [
    col
    for col in numeric_cols
    if col not in EXCLUDE_COLUMNS
]

print(
    f"\nNumerical features selected: "
    f"{len(feature_cols)}"
)

for col in feature_cols:
    print(f"  - {col}")

X_raw = df[feature_cols].copy()


# ============================================================
# 2. MEDIAN IMPUTATION
# ============================================================

print("\nMissing values before imputation:")

missing = X_raw.isna().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("  None")
else:
    print(missing)

imputer = SimpleImputer(
    strategy="median"
)

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
# 3. STANDARDIZE
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_imputed
)

n_samples, n_features = X_scaled.shape

print(
    f"\nData ready: "
    f"{n_samples:,} rows x "
    f"{n_features} standardized features"
)

print(
    f"Mean: {X_scaled.mean():.4f}"
)

print(
    f"Std:  {X_scaled.std():.4f}"
)


# ============================================================
# 4. FULL PCA
# ============================================================

print("\n" + "=" * 65)
print("FULL PCA")
print("=" * 65)

pca_full = PCA(
    n_components=n_features
)

pca_full.fit(X_scaled)

explained = (
    pca_full.explained_variance_ratio_
)

cumulative = np.cumsum(
    explained
)


# ============================================================
# 5. EXPLAINED VARIANCE
# ============================================================

print("\nExplained variance ratio:")

for i, (ev, cum) in enumerate(
    zip(explained, cumulative),
    start=1
):

    print(
        f"PC{i:2d}: "
        f"{ev:.4f}   "
        f"cumulative = {cum:.4f}"
    )


# Components required for 90% and 95%
n_90 = int(
    np.argmax(cumulative >= 0.90) + 1
)

n_95 = int(
    np.argmax(cumulative >= 0.95) + 1
)

print(
    f"\nComponents needed for >=90% variance: "
    f"{n_90}"
)

print(
    f"Components needed for >=95% variance: "
    f"{n_95}"
)


# ============================================================
# 6. SCREE + CUMULATIVE VARIANCE
# ============================================================

components = np.arange(
    1,
    n_features + 1
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)


# ----------------------------
# Scree plot
# ----------------------------

axes[0].bar(
    components,
    explained
)

axes[0].plot(
    components,
    explained,
    marker="o"
)

axes[0].set_title(
    "Scree Plot"
)

axes[0].set_xlabel(
    "Principal Component"
)

axes[0].set_ylabel(
    "Explained Variance Ratio"
)

axes[0].set_xticks(
    components
)

axes[0].grid(
    axis="y",
    alpha=0.3
)


# ----------------------------
# Cumulative variance
# ----------------------------

axes[1].plot(
    components,
    cumulative,
    marker="o"
)

axes[1].axhline(
    0.90,
    linestyle="--",
    label="90% threshold"
)

axes[1].axhline(
    0.95,
    linestyle="--",
    label="95% threshold"
)

axes[1].axvline(
    n_90,
    linestyle=":"
)

axes[1].axvline(
    n_95,
    linestyle=":"
)

axes[1].scatter(
    [n_90],
    [cumulative[n_90 - 1]],
    s=60,
    label=f"{n_90} PCs → 90%"
)

axes[1].scatter(
    [n_95],
    [cumulative[n_95 - 1]],
    s=60,
    label=f"{n_95} PCs → 95%"
)

axes[1].set_title(
    "Cumulative Explained Variance"
)

axes[1].set_xlabel(
    "Number of Components"
)

axes[1].set_ylabel(
    "Cumulative Explained Variance"
)

axes[1].set_xticks(
    components
)

axes[1].set_ylim(
    0,
    1.03
)

axes[1].grid(
    alpha=0.3
)

axes[1].legend()


plt.tight_layout()

scree_path = os.path.join(
    PLOT_DIR,
    "pca_scree_cumulative.png"
)

plt.savefig(
    scree_path,
    dpi=150
)

plt.close()

print(
    f"\nSaved plot -> {scree_path}"
)


# ============================================================
# 7. 2D PCA
# ============================================================

pca_2d = PCA(
    n_components=2
)

X_pca_2d = pca_2d.fit_transform(
    X_scaled
)

variance_2d = (
    pca_2d.explained_variance_ratio_.sum()
)

print(
    f"\n2D PCA explains "
    f"{variance_2d:.4f} "
    f"of total variance"
)

print(
    f"PC1 = "
    f"{pca_2d.explained_variance_ratio_[0]:.4f}"
)

print(
    f"PC2 = "
    f"{pca_2d.explained_variance_ratio_[1]:.4f}"
)


# ============================================================
# 8. 3D PCA
# ============================================================

pca_3d = PCA(
    n_components=3
)

X_pca_3d = pca_3d.fit_transform(
    X_scaled
)

variance_3d = (
    pca_3d.explained_variance_ratio_.sum()
)

print(
    f"\n3D PCA explains "
    f"{variance_3d:.4f} "
    f"of total variance"
)


# ============================================================
# 9. VISUALIZE 2D + 3D PCA
# ============================================================

placement_labels = (
    df["PlacementStatus"].values
)

fig = plt.figure(
    figsize=(13, 5)
)


# ----------------------------
# 2D
# ----------------------------

ax1 = fig.add_subplot(
    1,
    2,
    1
)

# Keep the faculty's visual distinction:
# placed vs not placed.
colors = np.where(
    placement_labels == 1,
    "black",
    "lightgray"
)

ax1.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=colors,
    s=4,
    alpha=0.35
)

ax1.set_title(
    f"2D PCA Projection "
    f"(PC1 + PC2 = "
    f"{variance_2d * 100:.1f}% variance)"
)

ax1.set_xlabel(
    f"PC1 "
    f"({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)"
)

ax1.set_ylabel(
    f"PC2 "
    f"({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)"
)

ax1.grid(
    alpha=0.3
)


# ----------------------------
# 3D
# ----------------------------

ax2 = fig.add_subplot(
    1,
    2,
    2,
    projection="3d"
)

ax2.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=colors,
    s=4,
    alpha=0.3
)

ax2.set_title(
    f"3D PCA Projection "
    f"(PC1 + PC2 + PC3 = "
    f"{variance_3d * 100:.1f}% variance)"
)

ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_zlabel("PC3")


plt.tight_layout()

projection_path = os.path.join(
    PLOT_DIR,
    "pca_2d_3d_projection.png"
)

plt.savefig(
    projection_path,
    dpi=150
)

plt.close()

print(
    f"\nSaved plot -> {projection_path}"
)


# ============================================================
# 10. SAVE 2D PCA
# ============================================================

pca2d_npy = os.path.join(
    MODEL_DIR,
    "X_pca2d.npy"
)

pca2d_csv = os.path.join(
    MODEL_DIR,
    "X_pca2d.csv"
)

np.save(
    pca2d_npy,
    X_pca_2d
)

np.savetxt(
    pca2d_csv,
    X_pca_2d,
    delimiter=",",
    header="PC1,PC2",
    comments="",
    fmt="%.6f"
)

print(
    f"\nSaved -> {pca2d_npy}"
)

print(
    f"Shape: {X_pca_2d.shape}"
)


# ============================================================
# 11. SAVE 10-COMPONENT PCA
# ============================================================

pca_10d = PCA(
    n_components=10
)

X_pca_10d = pca_10d.fit_transform(
    X_scaled
)

pca10d_npy = os.path.join(
    MODEL_DIR,
    "X_pca10d.npy"
)

pca10d_csv = os.path.join(
    MODEL_DIR,
    "X_pca10d.csv"
)

np.save(
    pca10d_npy,
    X_pca_10d
)

header_10d = ",".join(
    [
        f"PC{i + 1}"
        for i in range(
            X_pca_10d.shape[1]
        )
    ]
)

np.savetxt(
    pca10d_csv,
    X_pca_10d,
    delimiter=",",
    header=header_10d,
    comments="",
    fmt="%.6f"
)

variance_10d = (
    pca_10d.explained_variance_ratio_.sum()
)

print(
    f"\nSaved -> {pca10d_npy}"
)

print(
    f"Shape: {X_pca_10d.shape}"
)

print(
    f"10 components explain "
    f"{variance_10d * 100:.2f}% "
    f"of total variance"
)


# ============================================================
# 12. SAVE PCA METADATA
# ============================================================

metadata = {
    "n_samples": int(n_samples),
    "n_features": int(n_features),
    "feature_columns": feature_cols,
    "components_for_90_percent": n_90,
    "components_for_95_percent": n_95,
    "variance_2d": float(variance_2d),
    "variance_3d": float(variance_3d),
    "variance_10d": float(variance_10d),
    "random_state": RANDOM_STATE
}

metadata_path = os.path.join(
    MODEL_DIR,
    "pca_metadata.json"
)

with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 65)
print("PCA ANALYSIS COMPLETE")
print("=" * 65)

print(
    f"90% variance : {n_90} components"
)

print(
    f"95% variance : {n_95} components"
)

print(
    f"\nOutputs saved in:"
)

print(
    f"  {PLOT_DIR}"
)

print(
    f"  {MODEL_DIR}"
)