import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=== PlacementPredict EDA ===")

# --------------------------------------------------
# 1. Load raw dataset
# --------------------------------------------------

DATA_PATH = "Data/placement_predict_Dataset.csv"
PLOT_DIR = "EDA/plots"

os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Basic inspection
# --------------------------------------------------

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- First 5 Rows ---")
print(df.head())


# --------------------------------------------------
# 3. Missing-value analysis
# --------------------------------------------------

print("\n--- Missing Values ---")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    print(missing)

    plt.figure(figsize=(10, 6))
    missing.plot(kind="bar")
    plt.title("Missing Values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "missing_values.png"),
        dpi=150
    )
    plt.close()


# --------------------------------------------------
# 4. Target distribution
# --------------------------------------------------

print("\n--- Placement Status Distribution ---")
print(df["PlacementStatus"].value_counts())

print("\n--- Placement Status Percentage ---")
print(
    df["PlacementStatus"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="PlacementStatus")
plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "placement_distribution.png"),
    dpi=150
)
plt.close()


# --------------------------------------------------
# 5. Numerical features vs PlacementStatus
# --------------------------------------------------

print("\n--- Numerical Features vs PlacementStatus ---")

numeric_features = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]

# Keep only columns that actually exist
numeric_features = [
    col for col in numeric_features
    if col in df.columns
]

# Mean comparison
mean_comparison = (
    df.groupby("PlacementStatus")[numeric_features]
    .mean()
    .T
)

print("\nMean values by PlacementStatus:")
print(mean_comparison)

# Save numerical comparison
mean_comparison.to_csv(
    os.path.join(PLOT_DIR, "numeric_feature_comparison.csv")
)


# --------------------------------------------------
# 6. Important numerical feature plots
# --------------------------------------------------

important_features = [
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "CodingTestScore",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "MockInterviewScore"
]

important_features = [
    col for col in important_features
    if col in df.columns
]

for feature in important_features:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="PlacementStatus",
        y=feature
    )

    plt.title(f"{feature} vs Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel(feature)
    plt.tight_layout()

    filename = (
        feature.lower()
        .replace(" ", "_")
        + "_vs_placement.png"
    )

    plt.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=150
    )

    plt.close()


# --------------------------------------------------
# 7. Categorical features vs PlacementStatus
# --------------------------------------------------

print("\n--- Categorical Features vs PlacementStatus ---")

categorical_features = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Do not treat target as a feature
categorical_features = [
    col for col in categorical_features
    if col != "PlacementStatus"
]

if categorical_features:
    for feature in categorical_features:

        print(f"\n{feature}:")
        print(
            pd.crosstab(
                df[feature],
                df["PlacementStatus"],
                normalize="index"
            ).round(3)
        )

        plt.figure(figsize=(8, 5))

        sns.countplot(
            data=df,
            x=feature,
            hue="PlacementStatus"
        )

        plt.title(f"{feature} vs Placement Status")
        plt.xlabel(feature)
        plt.ylabel("Number of Students")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = (
            feature.lower()
            .replace(" ", "_")
            + "_vs_placement.png"
        )

        plt.savefig(
            os.path.join(PLOT_DIR, filename),
            dpi=150
        )

        plt.close()
else:
    print("No categorical feature columns found.")


# --------------------------------------------------
# 8. CGPA comparison
# --------------------------------------------------

print("\n--- Average CGPA by Placement Status ---")

cgpa_comparison = (
    df.groupby("PlacementStatus")["CGPA"]
    .mean()
)

print(cgpa_comparison)

plt.figure(figsize=(6, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="CGPA"
)

plt.title("CGPA vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "cgpa_vs_placement.png"),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 9. Correlation analysis
# --------------------------------------------------

print("\n--- Correlation Analysis ---")

# Select numeric columns
numeric_df = df.select_dtypes(
    include=[np.number]
).copy()

# Remove identifier and leakage column
columns_to_drop = [
    "StudentID",
    "Salary Package"
]

columns_to_drop = [
    col for col in columns_to_drop
    if col in numeric_df.columns
]

numeric_df = numeric_df.drop(
    columns=columns_to_drop
)

corr = numeric_df.corr()

print("\nCorrelation with PlacementStatus:")

if "PlacementStatus" in corr.columns:
    placement_corr = (
        corr["PlacementStatus"]
        .drop("PlacementStatus")
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print(placement_corr)


# Heatmap
plt.figure(figsize=(14, 11))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "correlation_heatmap.png"),
    dpi=150
)

plt.close()


# --------------------------------------------------
# 10. Statistical summary
# --------------------------------------------------

print("\n--- Statistical Summary ---")

print(df.describe())


# --------------------------------------------------
# 11. EDA Summary
# --------------------------------------------------

print("\n=== EDA Complete ===")

print("\nGenerated plots are stored in:")
print(PLOT_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(PLOT_DIR)):
    print(" -", file)

print("\nScript finished successfully.")