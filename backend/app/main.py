# Step 3 work by Mishi
# wired up to call an external train script via subprocess (simulated)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import model_utils
import os
import subprocess

app = FastAPI(title='Text Classifier API - step3')

class TextIn(BaseModel):
    text: str

@app.post('/predict/scratch')
async def predict_scratch(payload: TextIn):
    try:
        return model_utils.predict_scratch(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/train')
async def train():
    # simulate calling train.py located one level up
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_script = os.path.join(backend_root, 'train.py')
    try:
        proc = subprocess.run(['python', train_script], cwd=backend_root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        return {'status': 'ok', 'detail': proc.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Text Classifier API (step3)'}
