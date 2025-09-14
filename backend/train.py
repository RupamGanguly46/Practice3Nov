# Step 5 work by Rupam
# Minor refactor: extracted build_pipeline function
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scratch_model.joblib")


def build_pipeline(ngram=(1,2), max_features=4500, random_state=42):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=ngram, max_features=max_features)),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
    ])

def train_and_save(random_state=42, ngram=(1,2)):
    print("[Rupam step5] Building pipeline and training")
    df = pd.read_csv(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label'].astype(str)
    pipe = build_pipeline(ngram=ngram, max_features=4500, random_state=random_state)
    pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved scratch model to {MODEL_PATH}")
    return MODEL_PATH

if __name__ == '__main__':
    train_and_save()
