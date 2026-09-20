from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100000
    )