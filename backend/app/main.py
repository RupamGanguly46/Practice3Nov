# Step 2 work by Mishi
# Added endpoints for predict and train
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import model_utils
import os

app = FastAPI(title='Text Classifier API - step2')

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
    return {'status': 'ok', 'detail': 'train triggered (simulated)'}

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Text Classifier API (step2)'}
