import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "placement_predict_cleaned.csv"
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("=== Feature / Target Separation ===")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# Target and excluded columns
# --------------------------------------------------

TARGET_COL = "PlacementStatus"
ID_COL = "StudentID"

# Salary Package MUST NOT be used to predict placement.
# It is only known after placement and would cause data leakage.
EXCLUDE_COLS = [
    ID_COL,
    TARGET_COL,
    "Salary Package"
]

# --------------------------------------------------
# Separate X and y
# --------------------------------------------------

X = df.drop(columns=EXCLUDE_COLS)
y = df[TARGET_COL]

# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
for feature in X.columns:
    print("-", feature)

print("\nClass distribution:")
print(y.value_counts())

print("\nClass distribution (%):")
print(
    y.value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n=== Feature / Target Separation Complete ===")