from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__, template_folder="app/templates")

# Load the trained DistilBERT model
classifier = pipeline(
    "sentiment-analysis",
    model="./models/distilbert_sentiment",
    tokenizer="./models/distilbert_sentiment"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({
            "error": "Please provide 'text' in the request body"
        }), 400

    text = data["text"]

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "'text' must be a non-empty string"
        }), 400

    result = classifier(text)[0]

    label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
    }

    return jsonify({
        "text": text,
        "sentiment": label_map[result["label"]],
        "confidence": round(result["score"], 4)
    })


if __name__ == "__main__":
    app.run(debug=True)