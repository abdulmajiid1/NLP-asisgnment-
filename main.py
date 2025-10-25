# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path

# load the saved model and vectorizer
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
VEC_PATH = MODELS_DIR / "vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

app = FastAPI(title="Tweet Moderation API", version="1.0")

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class PredictResponse(BaseModel):
    label: str
    label_id: int
    prob_foul: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = vectorizer.transform([req.text])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[:, 1][0])
    else:
        prob = float(model.decision_function(X)[0])
    label_id = 1 if prob >= 0.5 else 0
    label = "foul" if label_id == 1 else "proper"
    return PredictResponse(label=label, label_id=label_id, prob_foul=prob)
