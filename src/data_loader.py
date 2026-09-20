import pandas as pd
from preprocessing import clean_text

DATA_PATH = "data/raw/imdb.csv"
OUTPUT_PATH = "data/processed/cleaned_dataset.csv"


def load_and_process_data():

    # Load raw dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Remove duplicate reviews
    duplicate_count = df["review"].duplicated().sum()
    print("Duplicate reviews found:", duplicate_count)

    df = df.drop_duplicates(subset="review").reset_index(drop=True)

    print("Dataset shape after removing duplicates:", df.shape)

    # Clean reviews
    print("\nCleaning reviews...")

    df["cleaned_review"] = df["review"].apply(clean_text)

    # Save processed dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("Cleaning completed!")
    print("Processed dataset saved to:", OUTPUT_PATH)

    # Show sample
    print("\nSample:")
    print(df[["review", "cleaned_review", "sentiment"]].head())

    return df


if __name__ == "__main__":
    load_and_process_data()