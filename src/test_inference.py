from inference import predict_sentiment


reviews = [
    "This movie was absolutely fantastic. I loved every minute of it.",
    "This movie was terrible and extremely boring.",
    "I really enjoyed this film.",
    "I would not recommend this movie to anyone."
]


for review in reviews:

    result = predict_sentiment(review)

    print("\nReview:", review)
    print("Result:", result)