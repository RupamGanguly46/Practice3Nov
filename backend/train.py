# Step 8 work by Rupam
# Final step: small docstring improvements and default param changes
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os
import json

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scratch_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "scratch_model_meta.json")


def build_pipeline(ngram=(1,2), max_features=4500, random_state=42):
    """Return a scikit-learn pipeline ready for training."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram, max_features=max_features)),
        ('clf', LogisticRegression(max_iter=1200, random_state=random_state))
    ])


def train_and_save(random_state=7, ngram=(1,2)):
    """Train the pipeline and save model and metadata."""
    print("[Rupam step8] Training (final snapshot) with random_state=7")
    df = pd.read_csv(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label'].astype(str)
    pipe = build_pipeline(ngram=ngram, max_features=4500, random_state=random_state)
    pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    meta = {"trained_on_rows": len(df), "ngram": ngram, "version_note": "step8"}
    with open(META_PATH, 'w') as fh:
        json.dump(meta, fh)
    print(f"Saved scratch model to {MODEL_PATH} and meta to {META_PATH}")
    return MODEL_PATH

if __name__ == '__main__':
    train_and_save()
