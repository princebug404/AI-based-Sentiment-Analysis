import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


DATA_PATH = "data/processed/cleaned_dataset.csv"


def prepare_lstm_data():

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    X = df["cleaned_review"]
    y = df["sentiment"]

    # Convert labels to numbers
    y = y.map({
        "negative": 0,
        "positive": 1
    })

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create tokenizer
    tokenizer = Tokenizer(
        num_words=30000,
        oov_token="<OOV>"
    )

    # Learn vocabulary only from training data
    tokenizer.fit_on_texts(X_train)

    # Convert text to integer sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    # Padding
    max_length = 300

    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    print("LSTM preprocessing completed!")

    print("\nTraining samples:", X_train_padded.shape[0])
    print("Testing samples:", X_test_padded.shape[0])

    print("\nVocabulary size:", len(tokenizer.word_index))

    print("\nTraining sequence shape:", X_train_padded.shape)
    print("Testing sequence shape:", X_test_padded.shape)

    print("\nExample sequence:")
    print(X_train_padded[0])

    return (
        X_train_padded,
        X_test_padded,
        y_train,
        y_test,
        tokenizer
    )


if __name__ == "__main__":
    prepare_lstm_data()