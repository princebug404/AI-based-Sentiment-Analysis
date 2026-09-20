import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_model


DATA_PATH = "data/processed/cleaned_dataset.csv"


def train_logistic_regression():

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

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(
        max_iter=1000
    )

    print("Training Logistic Regression...")

    model.fit(X_train_tfidf, y_train)

    # Predictions
    print("Making predictions...")

    y_pred = model.predict(X_test_tfidf)

    # Evaluation
    results = evaluate_model(y_test, y_pred)

    return model, vectorizer, results


if __name__ == "__main__":
    train_logistic_regression()