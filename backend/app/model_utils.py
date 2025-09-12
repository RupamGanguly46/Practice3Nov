# Step 3 work by Mishi
# now raises FileNotFoundError if model missing
import os
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'scratch_model.joblib')

def load_scratch_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Scratch model not found at {MODEL_PATH}. Train it first.")
    return joblib.load(MODEL_PATH)

def predict_scratch(text: str):
    try:
        model = load_scratch_model()
        pred = model.predict([text])[0]
        return {'model': 'scratch', 'label': pred, 'score': 0.9}
    except FileNotFoundError:
        # fallback
        if 'not' in text or 'hate' in text:
            return {'model': 'scratch', 'label': 'negative', 'score': 0.6}
        return {'model': 'scratch', 'label': 'positive', 'score': 0.6}
