# Step 2 work by Mishi
# model_utils with scratch predict returning classes
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

def predict_scratch(text: str):
    # naive heuristic
    if 'not' in text or 'hate' in text:
        return {'model': 'scratch', 'label': 'negative', 'score': 0.9}
    return {'model': 'scratch', 'label': 'positive', 'score': 0.9}
