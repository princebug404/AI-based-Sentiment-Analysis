import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/cleaned_dataset.csv"


def split_dataset():

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    X = df["cleaned_review"]
    y = df["sentiment"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Dataset split successfully!")

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nTesting label distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    split_dataset()