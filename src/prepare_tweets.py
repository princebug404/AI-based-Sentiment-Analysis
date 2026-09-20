import pandas as pd


INPUT_FILE = "data/raw/Tweets.csv"
OUTPUT_FILE = "data/raw/tweets_clean.csv"


df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("INITIAL DATASET")
print("=" * 60)
print(f"Rows: {len(df)}")


# Keep only the columns needed for sentiment classification
df = df[["text", "airline_sentiment", "airline_sentiment_confidence"]].copy()


# Remove missing text/labels
df = df.dropna(subset=["text", "airline_sentiment"])


# Normalize labels
df["airline_sentiment"] = df["airline_sentiment"].str.lower().str.strip()


# Remove exact duplicate texts
before = len(df)

df = df.drop_duplicates(subset=["text"])

after = len(df)

print(f"\nDuplicate texts removed: {before - after}")


# Keep only our three target classes
valid_labels = {
    "negative",
    "neutral",
    "positive"
}

df = df[df["airline_sentiment"].isin(valid_labels)]


# Rename columns
df = df.rename(
    columns={
        "airline_sentiment": "sentiment",
        "airline_sentiment_confidence": "annotation_confidence"
    }
)


print("\n" + "=" * 60)
print("CLEAN DATASET")
print("=" * 60)

print(f"Rows: {len(df)}")

print("\nClass distribution:")
print(df["sentiment"].value_counts())

print("\nClass percentages:")
print(
    (df["sentiment"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(df.isnull().sum())

print("\nText length statistics (words):")

word_lengths = df["text"].str.split().str.len()

print(word_lengths.describe())

print("\nLongest text:")
longest_index = word_lengths.idxmax()
print(df.loc[longest_index, "text"])
print(f"Words: {word_lengths.loc[longest_index]}")


# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")