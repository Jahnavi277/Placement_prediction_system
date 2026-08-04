print("Script started")

import pandas as pd

df = pd.read_csv("Data/placement_predict_Dataset.csv")
print(df.head())

print("Script finished")

print(df.info())
print(df.describe())
print(df.isnull().sum())