import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

DATA_PATH = "data/processed/cleaned_dataset.csv"

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256


def train_transformer():

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Convert labels
    df["label"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    # Keep only required columns
    df = df[["cleaned_review", "label"]]

    # Train/Test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(
        train_df,
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test_df,
        preserve_index=False
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize_function(examples):

        return tokenizer(
            examples["cleaned_review"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("\nTokenizing dataset...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    # Remove original text column
    train_dataset = train_dataset.remove_columns(
        ["cleaned_review"]
    )

    test_dataset = test_dataset.remove_columns(
        ["cleaned_review"]
    )

    # Load pretrained model
    print("\nLoading DistilBERT...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    # Training arguments
    training_args = TrainingArguments(
    output_dir="./models/distilbert",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    fp16=True,

    report_to="none"
)

    # Metrics
    def compute_metrics(eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1
        )

        return {
            "accuracy": accuracy_score(
                labels,
                predictions
            ),

            "precision": precision_score(
                labels,
                predictions
            ),

            "recall": recall_score(
                labels,
                predictions
            ),

            "f1": f1_score(
                labels,
                predictions
            )
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    print("\n===== Training DistilBERT =====")

    trainer.train()

    # Evaluate
    print("\n===== DistilBERT Evaluation =====")

    results = trainer.evaluate()

    print(results)

    # Predictions
    predictions = trainer.predict(
        test_dataset
    )

    y_pred = np.argmax(
        predictions.predictions,
        axis=-1
    )

    y_test = predictions.label_ids

    # Final metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n===== Final DistilBERT Results =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    MODEL_DIR = "models/distilbert_sentiment"

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"\nModel and tokenizer saved to: {MODEL_DIR}")

    return trainer


if __name__ == "__main__":
    train_transformer()