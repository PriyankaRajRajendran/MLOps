# Wine Quality Prediction - Streamlit ML Application

## Project Overview
A machine learning web application built with Streamlit that predicts wine quality based on chemical properties. Uses Random Forest classification with 86% accuracy.

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML Model**: Random Forest (Scikit-learn)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## Installation

### 1. Clone/Create Project
```bash
mkdir wine-quality-app
cd wine-quality-app
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install scikit-learn pandas numpy fastapi uvicorn streamlit matplotlib seaborn requests
```

## Project Files
- `Dashboard.py` - Main Streamlit application with 4-tab interface
- `main.py` - FastAPI backend for model serving
- `train.py` - Model training script
- `wine_quality_merged.csv` - Dataset (6,497 samples)
- `wine_model.pkl` - Trained model (generated)

## Running the Application

### Step 1: Train the Model
```bash
python train.py
```
Creates `wine_model.pkl` with 85.8% accuracy

### Step 2: Start Backend (Terminal 1)
```bash
uvicorn main:app --reload
```
Runs on http://localhost:8000

### Step 3: Start Streamlit (Terminal 2)
```bash
streamlit run Dashboard.py
```
Opens at http://localhost:8501

### Our Enhanced Wine App
- **Dataset**: Wine quality (11 chemical features + type)
- **Interface**: 4-tab professional dashboard
- **Model**: Binary classification with confidence scores
- **Features**: Analytics, learning content, batch processing
- **UI**: Custom styling, charts, interactive visualizations

- 11 chemical property sliders 
- Wine type selector (Red/White)
- **Quick Presets** (Premium/Average) 
- Real-time wine profile display 
- Confidence meter with progress bar 
- Quality improvement suggestions 


## Application Features

### Predict Tab
- 11 chemical property sliders
- Red/White wine selection
- Premium/Average presets
- Real-time predictions with confidence scores

### Analytics Tab
- Prediction history tracking
- Quality distribution charts
- Confidence metrics

###  Learn Tab
- Wine chemistry explanations
- Feature importance visualization
- Quality scale information

###  Batch Predict Tab
- CSV file upload for bulk predictions
- Sample template download

## Model Performance
- **Algorithm**: Random Forest 
- **Accuracy**: 85.8% 
- **Dataset**: 6,497 samples 
- **Features**: 11 chemical + 1 type 
- **Top 3 Important Features**:
  1. Alcohol (21.1%)
  2. Density (13.1%)
  3. Volatile Acidity (9.3%)


## Dataset Features
| Feature | Range | Unit |
|---------|-------|------|
| Fixed Acidity | 3.0-15.0 | g/L |
| Volatile Acidity | 0.0-2.0 | g/L |
| Citric Acid | 0.0-2.0 | g/L |
| Residual Sugar | 0.0-20.0 | g/L |
| Chlorides | 0.0-0.5 | g/L |
| Free SO₂ | 0-100 | mg/L |
| Total SO₂ | 0-300 | mg/L |
| Density | 0.98-1.04 | g/mL |
| pH | 2.5-4.5 | - |
| Sulphates | 0.0-2.0 | g/L |
| Alcohol | 8.0-15.0 | % |

## Author
Priyanka Raj 
Mlops Labs
