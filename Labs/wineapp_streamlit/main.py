from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Wine Quality Prediction API")

# Define input schema
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    type: str  # "red" or "white"

# Load the model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        with open('wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model not found. Please run train.py first to create wine_model.pkl")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Wine Quality Prediction API is running",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_wine_quality(wine: WineFeatures):
    """Predict wine quality based on chemical properties"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Prepare input data
        features = pd.DataFrame([{
            'fixed acidity': wine.fixed_acidity,
            'volatile acidity': wine.volatile_acidity,
            'citric acid': wine.citric_acid,
            'residual sugar': wine.residual_sugar,
            'chlorides': wine.chlorides,
            'free sulfur dioxide': wine.free_sulfur_dioxide,
            'total sulfur dioxide': wine.total_sulfur_dioxide,
            'density': wine.density,
            'pH': wine.pH,
            'sulphates': wine.sulphates,
            'alcohol': wine.alcohol,
            'type': 1 if wine.type == "red" else 0  # Encode type
        }])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        
        return {
            "quality": int(prediction),  # 1 for high quality, 0 for standard
            "quality_label": "High Quality" if prediction == 1 else "Standard Quality",
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")