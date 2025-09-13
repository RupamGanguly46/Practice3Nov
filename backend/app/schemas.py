# Step 4 work by Mishi
from pydantic import BaseModel

class TextIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    model: str
    label: str
    score: float
