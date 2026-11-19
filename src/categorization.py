"""Train an expense classifier from fictional, labeled merchant descriptions."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA = ROOT / "data" / "synthetic_training.csv"
OTHER = "Other"


def train_classifier(training_path=TRAINING_DATA):
    """Fit vectorizer and classifier together so prediction uses the same vocabulary."""
    examples = pd.read_csv(training_path, keep_default_na=False)
    model = Pipeline([
        # Frequent terms that appear in few descriptions receive more weight.
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=500)),
    ])
    return model.fit(examples["description"], examples["category"])


def categorize(transactions, model):
    """Return a copy with expense labels and the model's top-class probabilities."""
    result = transactions.assign(category="No activity", confidence=np.nan)
    result.loc[result["amount"] > 0, "category"] = "Credit"
    expenses = result["amount"] < 0
    if not expenses.any():
        return result

    descriptions = result.loc[expenses, "description"]
    vectors = model.named_steps["tfidf"].transform(descriptions)
    classifier = model.named_steps["classifier"]
    probabilities = classifier.predict_proba(vectors)
    has_terms = vectors.getnnz(axis=1) > 0
    labels = np.where(has_terms, classifier.classes_[probabilities.argmax(axis=1)], OTHER)
    result.loc[expenses, "category"] = labels
    result.loc[expenses, "confidence"] = np.where(has_terms, probabilities.max(axis=1), np.nan)
    return result


if __name__ == "__main__":
    # The artifact contains a model trained exclusively on the bundled synthetic data.
    destination = ROOT / "models" / "expense_classifier.joblib"
    destination.parent.mkdir(exist_ok=True)
    joblib.dump(train_classifier(), destination)
    print(f"Saved {destination.relative_to(ROOT)}")
