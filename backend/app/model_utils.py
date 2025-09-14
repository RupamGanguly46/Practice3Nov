# Step 5 work by Mishi
# Slight refactor and caching of loaded models
import os
import joblib
from transformers import pipeline

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'scratch_model.joblib')

_scr_model = None
_hf_pipeline = None

def load_scratch_model():
    global _scr_model
    if _scr_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Scratch model not found at {MODEL_PATH}. Train it first.")
        _scr_model = joblib.load(MODEL_PATH)
    return _scr_model


def predict_scratch(text: str):
    model = load_scratch_model()
    pred = model.predict_proba([text])[0]
    labels = model.classes_
    best_idx = pred.argmax()
    return {
        'model': 'scratch',
        'label': labels[best_idx],
        'score': float(pred[best_idx])
    }


def load_hf_pipeline():
    global _hf_pipeline
    if _hf_pipeline is None:
        _hf_pipeline = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', return_all_scores=False)
    return _hf_pipeline


def predict_hf(text: str):
    pipe = load_hf_pipeline()
    res = pipe(text)[0]
    return {
        'model': 'huggingface',
        'label': res['label'].lower(),
        'score': float(res['score'])
    }
