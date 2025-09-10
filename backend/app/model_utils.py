# Step 1 work by Mishi
# Minimal model utils placeholder
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

def predict_scratch(text: str):
    return {'model': 'scratch', 'label': 'neutral', 'score': 0.5}
