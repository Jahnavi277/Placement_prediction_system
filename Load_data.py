import pandas as pd

print("=== PlacementPredict Data Cleaning ===")

# 1. Load raw dataset
input_file = "Data/placement_predict_Dataset.csv"
output_file = "Data/placement_predict_cleaned.csv"

df = pd.read_csv(input_file)

print("\nOriginal dataset shape:", df.shape)

# 2. Show original missing values
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 3. Remove duplicate rows
duplicates = df.duplicated().sum()
print("\nDuplicate rows found:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# 4. Handle missing numerical values
# These are the columns identified during EDA
numeric_columns = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]

print("\n--- Filling Missing Numerical Values ---")

for column in numeric_columns:
    missing = df[column].isnull().sum()

    if missing > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

        print(
            f"{column}: {missing} missing values "
            f"filled with median {median_value:.2f}"
        )

# 5. Validate SGPA values
sgpa_columns = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8"
]

for column in sgpa_columns:
    invalid = (df[column] < 0) | (df[column] > 10)
    count = invalid.sum()

    if count > 0:
        print(f"{column}: {count} invalid values found")
        df.loc[invalid, column] = pd.NA

# 6. Validate CGPA
invalid_cgpa = (df["CGPA"] < 0) | (df["CGPA"] > 10)

if invalid_cgpa.sum() > 0:
    print("CGPA:", invalid_cgpa.sum(), "invalid values found")
    df.loc[invalid_cgpa, "CGPA"] = pd.NA

# 7. Validate attendance
invalid_attendance = (
    (df["AttendancePercent"] < 0) |
    (df["AttendancePercent"] > 100)
)

if invalid_attendance.sum() > 0:
    print(
        "AttendancePercent:",
        invalid_attendance.sum(),
        "invalid values found"
    )

    df.loc[invalid_attendance, "AttendancePercent"] = pd.NA

# 8. Validate count-based columns
count_columns = [
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications"
]

for column in count_columns:
    invalid = df[column] < 0
    count = invalid.sum()

    if count > 0:
        print(f"{column}: {count} negative values found")
        df.loc[invalid, column] = pd.NA

# 9. Fill missing values created by invalid-value checks
for column in df.select_dtypes(include="number").columns:

    missing = df[column].isnull().sum()

    if missing > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

# 10. Final missing-value check
print("\n--- Missing Values After Cleaning ---")

remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) == 0:
    print("No missing values remaining.")
else:
    print(remaining_missing)

# 11. Final duplicate check
print(
    "\nDuplicates after cleaning:",
    df.duplicated().sum()
)

# 12. Save cleaned dataset
df.to_csv(output_file, index=False)

print("\n=== Cleaning Complete ===")
print("Cleaned dataset saved to:")
print(output_file)

print("\nFinal dataset shape:", df.shape)

print("\nColumns:", len(df.columns))

print("\nScript finished successfully.")