# src/predict.py

from pydantic import BaseModel
from joblib import load
from pathlib import Path
import pandas as pd

MODEL_PATH = Path("model/wine_model.pkl")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float
    type: int

model = None

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found.")
    return load(MODEL_PATH)

def ensure_model_loaded():
    global model
    if model is None:
        model = load_model()

def predict_quality(input_data: WineInput) -> int:
    ensure_model_loaded()
    input_df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(input_df)[0]
    return int(round(float(prediction)))


