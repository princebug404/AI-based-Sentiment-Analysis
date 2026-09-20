import re


def clean_text(text):
    """
    Clean a single review.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags such as <br />
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    sample = """
    I absolutely LOVE this movie! <br /><br />
    It was amazing and worth watching.
    """

    print("Original:")
    print(sample)

    print("\nCleaned:")
    print(clean_text(sample))