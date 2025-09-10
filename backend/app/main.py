# Step 1 work by Mishi
# Basic FastAPI app skeleton and endpoint placeholders
from fastapi import FastAPI, HTTPException

app = FastAPI(title='Text Classifier API - step1')

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Text Classifier API (step1)'}
