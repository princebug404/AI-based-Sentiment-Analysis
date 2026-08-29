positive_words = ["good", "great", "excellent", "amazing", "love", "happy"]
negative_words = ["bad", "terrible", "awful", "hate", "worst", "sad"]


text = input("Enter a sentence: ")

words = text.lower().split()

positive_count = 0
negative_count = 0

for word in words:
    if word in positive_words:
        positive_count += 1

    if word in negative_words:
        negative_count += 1


if positive_count > negative_count:
    sentiment = "Positive"

elif negative_count > positive_count:
    sentiment = "Negative"

else:
    sentiment = "Neutral"


print("Sentiment:", sentiment)