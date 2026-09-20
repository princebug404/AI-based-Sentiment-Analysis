import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")


def tokenize_text(text):
    """
    Convert text into individual tokens.
    """
    return word_tokenize(text)


if __name__ == "__main__":

    sample = "This movie was absolutely amazing!"

    tokens = tokenize_text(sample)

    print("Original text:")
    print(sample)

    print("\nTokens:")
    print(tokens)