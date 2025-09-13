# Step 4 work by Mishi
# Add /predict/hf endpoint using model_utils.predict_hf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import model_utils
import os
import subprocess

app = FastAPI(title='Text Classifier API - step4')

class TextIn(BaseModel):
    text: str

@app.post('/predict/scratch')
async def predict_scratch(payload: TextIn):
    try:
        return model_utils.predict_scratch(payload.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/predict/hf')
async def predict_hf(payload: TextIn):
    try:
        return model_utils.predict_hf(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def train():
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_script = os.path.join(backend_root, 'train.py')
    proc = subprocess.run(['python', train_script], cwd=backend_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=proc.stderr)
    return {'status': 'ok', 'detail': proc.stdout}

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Text Classifier API (step4)'}
