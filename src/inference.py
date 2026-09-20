import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "models/distilbert_sentiment"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Set model to evaluation mode
model.eval()


def predict_sentiment(text):

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    # Get predicted class
    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    # Get confidence
    confidence = probabilities[0][predicted_class].item()

    label_map = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

    label = label_map[predicted_class]

    return {
        "sentiment": label,
        "confidence": round(confidence, 4)
    }