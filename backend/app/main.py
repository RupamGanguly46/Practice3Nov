# Step 6 work by Mishi
# Finalized main.py for Mishi: same as production snapshot with small comment
from fastapi import FastAPI, HTTPException
from app.schemas import TextIn, PredictOut
from app import model_utils
import os
import subprocess

app = FastAPI(title='Text Classifier API')

@app.post('/train')
async def train():
    try:
        backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        train_script = os.path.join(backend_root, 'train.py')
        proc = subprocess.run(['python', train_script], cwd=backend_root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        return {'status': 'ok', 'detail': proc.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/scratch', response_model=PredictOut)
async def predict_scratch(payload: TextIn):
    try:
        return model_utils.predict_scratch(payload.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/hf', response_model=PredictOut)
async def predict_hf(payload: TextIn):
    try:
        return model_utils.predict_hf(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Text Classifier API'}
# Step 6: finalized by Mishi
