# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI(title="Wine Quality Prediction API")

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

# Load model
model_path = os.path.join("model", "wine_model.pkl")
model = joblib.load(model_path)

@app.get("/features")
def list_features():
    return {
        "message": "Welcome to the Wine Quality Prediction API",
        "usage": "Use POST /predict with the fields below to get a quality score.",
        "fixed_acidity": "Range: 4 – 15.9",
        "volatile_acidity": "Range: 0.1 – 1.6",
        "citric_acid": "Range: 0.0 – 1.0",
        "residual_sugar": "Range: 0.6 – 65.8",
        "chlorides": "Range: 0.01 – 0.61",
        "free_sulfur_dioxide": "Range: 1 – 289",
        "total_sulfur_dioxide": "Range: 6 – 440",
        "density": "Range: 0.990 – 1.004",
        "ph": "Range: 2.7 – 4.0",
        "sulphates": "Range: 0.3 – 2.0",
        "alcohol": "Range: 8.0 – 14.9",
        "type": "0 = Red, 1 = White"
    },
    

@app.post("/predict")
def predict_wine(features: WineInput):
    input_df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(round(float(prediction)))}