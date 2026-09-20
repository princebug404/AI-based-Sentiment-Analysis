import pandas as pd


FILE_PATH = "data/raw/Tweets.csv"


df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Look for likely sentiment columns
for column in df.columns:
    if "sentiment" in column.lower():
        print(f"\nValue counts for '{column}':")
        print(df[column].value_counts(dropna=False))


# Text length analysis for likely text columns
for column in df.columns:
    if df[column].dtype == "object":
        lengths = df[column].astype(str).str.split().str.len()

        print(f"\nText-length statistics for '{column}':")
        print(lengths.describe())