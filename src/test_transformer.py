from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("Model loaded successfully!")

print("\nModel:", MODEL_NAME)
print("Number of labels:", model.config.num_labels)