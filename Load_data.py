import pandas as pd

print("Script started")

 #Load dataset
df = pd.read_csv("Data/placement_predict_Dataset.csv")

print("\n--- ORIGINAL DATA ---")
print(df.head())

 #1. Check shape
print("\nDataset shape:")
print(df.shape)

 #2. Check column names
print("\nColumns:")
print(df.columns)

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 5. Check data types
print("\nData types:")
print(df.dtypes)

# 6. Check basic statistics
print("\nStatistics:")
print(df.describe(include="all"))

print("\nScript finished")
# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Fill missing numerical values with median
numeric_columns = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# 4. Check for invalid values

# CGPA and SGPA should normally be between 0 and 10
sgpa_columns = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8"
]
for column in sgpa_columns:
    df.loc[(df[column] < 0) | (df[column] > 10), column] = pd.NA

df.loc[(df["CGPA"] < 0) | (df["CGPA"] > 10), "CGPA"] = pd.NA

# Attendance should be 0–100
df.loc[
    (df["AttendancePercent"] < 0) |
    (df["AttendancePercent"] > 100),
    "AttendancePercent"
] = pd.NA

# Count-based columns cannot be negative
count_columns = [
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications"
]

for column in count_columns:
    df.loc[df[column] < 0, column] = pd.NA

# 5. Fill any missing values created by invalid-value checks
for column in df.select_dtypes(include="number").columns:
    df[column] = df[column].fillna(df[column].median())

# 6. Check final missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 7. Check duplicates again
print("\nDuplicates after cleaning:", df.duplicated().sum())

# 8. Save cleaned dataset
df.to_csv("Data/placement_predict_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final shape:", df.shape)

print("\nScript finished")