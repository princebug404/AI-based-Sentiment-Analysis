from inference import predict_sentiment


class SentimentAgent:
    """
    Intelligent agent for sentiment analysis.

    PEAS:
    - Sensors: Incoming text
    - Environment: Text/review source
    - Actuators: Sentiment label and confidence
    - Performance Measure: F1-score
    """

    def perceive(self, text):
        """Receive and validate input from the environment."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        return text.strip()

    def decide(self, text):
        """Use the trained sentiment model to make a decision."""
        return predict_sentiment(text)

    def act(self, prediction):
        """Return the agent's decision to the user."""
        return prediction

    def run(self, text):
        """Execute the complete Perceive → Decide → Act cycle."""

        perceived_text = self.perceive(text)

        decision = self.decide(perceived_text)

        output = self.act(decision)

        return output


if __name__ == "__main__":

    agent = SentimentAgent()

    text = input("Enter a review: ")

    try:
        result = agent.run(text)

        print("\nAgent Output:")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")

    except ValueError as e:
        print(f"Error: {e}")