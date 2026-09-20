import pandas as pd


results = [
    {
        "Model": "Naive Bayes",
        "Accuracy": 0.8784,
        "Precision": 0.8730,
        "Recall": 0.8867,
        "F1-Score": 0.8798
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.9052,
        "Precision": 0.8948,
        "Recall": 0.9192,
        "F1-Score": 0.9068
    },
    {
        "Model": "SVM",
        "Accuracy": 0.9045,
        "Precision": 0.8987,
        "Recall": 0.9126,
        "F1-Score": 0.9056
    }
]


df = pd.DataFrame(results)

print("\n===== Model Comparison =====\n")
print(df.to_string(index=False))

print("\nBest model by F1-Score:")
best_model = df.loc[df["F1-Score"].idxmax()]

print(best_model["Model"])
print(f"F1-Score: {best_model['F1-Score']:.4f}")