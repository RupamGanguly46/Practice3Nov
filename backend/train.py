# Step 1 work by Rupam
# Initial training script (small change for step 1)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_train.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "scratch_model.joblib")

def train_and_save(random_state=42):
    df = pd.read_csv(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label'].astype(str)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000)),  # step1 uses 3000
        ('clf', LogisticRegression(max_iter=500, random_state=random_state))
    ])
    pipe.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved scratch model to {MODEL_PATH}")
    return MODEL_PATH

if __name__ == '__main__':
    train_and_save()
