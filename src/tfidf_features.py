import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/processed/cleaned_dataset.csv"


def create_tfidf_features():

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    X = df["cleaned_review"]
    y = df["sentiment"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    # Learn TF-IDF only from training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform test data using the already learned vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF pipeline completed!")

    print("\nTraining samples:", X_train_tfidf.shape[0])
    print("Testing samples:", X_test_tfidf.shape[0])

    print("\nNumber of features:", X_train_tfidf.shape[1])

    print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
    print("Testing TF-IDF shape:", X_test_tfidf.shape)

    return (
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        vectorizer
    )


if __name__ == "__main__":
    create_tfidf_features()