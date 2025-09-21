# Wine Quality Prediction API 🍷

A FastAPI-based ML service that predicts wine quality scores using Linear Regression.

---

## Quick Start 

**Install dependencies**

```
pip install -r requirements.txt
```
**To train the model**

```
python -m src.train
```

**Run the API**

```
uvicorn src.main:app --reload   
```
**Or**

```
python -m webbrowser http://127.0.0.1:8000/docs & uvicorn src.main:app --reload   
```

API Usage 
-----------

**POST** `/predict` - Predict wine quality

**Input Example:**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 34,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": 0
}
```

**Output:**
```json
{
  "predicted_quality": 5,
}
```