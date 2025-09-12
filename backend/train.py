# Step 3 work by Rupam
# Added argument to allow changing ngram_range via env var (simulated)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scratch_model.joblib")

def train_and_save(random_state=42, ngram=(1,2)):
    print("[Rupam step3] Training with ngram:", ngram)
    df = pd.read_csv(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label'].astype(str)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram, max_features=4000)),
        ('clf', LogisticRegression(max_iter=800, random_state=random_state))
    ])
    pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved scratch model to {MODEL_PATH}")
    return MODEL_PATH

if __name__ == '__main__':
    train_and_save()
