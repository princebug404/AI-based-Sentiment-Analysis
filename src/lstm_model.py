import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = "data/processed/cleaned_dataset.csv"

MAX_WORDS = 30000
MAX_LENGTH = 200


def train_lstm():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df["cleaned_review"]

    y = df["sentiment"].map({
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

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(X_train)

    # Convert text to sequences
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    # Padding
    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    print("LSTM data prepared!")

    print("\nTraining shape:", X_train_padded.shape)
    print("Testing shape:", X_test_padded.shape)

    # Build model
    model = Sequential([
    Embedding(
        input_dim=MAX_WORDS,
        output_dim=128
    ),

        LSTM(128),

        Dropout(0.5),

        Dense(1, activation="sigmoid")
    ])
    
    model.build(input_shape=(None, MAX_LENGTH))

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("\n===== Model Architecture =====")
    model.summary()

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # Train
    print("\n===== Training LSTM =====")

    history = model.fit(
        X_train_padded,
        y_train,
        epochs=8,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    # Predictions
    print("\nMaking predictions...")

    probabilities = model.predict(
        X_test_padded,
        batch_size=64
    )

    y_pred = (probabilities >= 0.5).astype(int).flatten()

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print("\n===== LSTM Results =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return model, tokenizer, history


if __name__ == "__main__":
    train_lstm()