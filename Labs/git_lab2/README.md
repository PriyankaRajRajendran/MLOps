# Wine Quality Prediction - MLOps Pipeline (Git_Lab2)

## Project Description
This project implements an automated machine learning pipeline using GitHub Actions for continuous integration and deployment. The pipeline automatically trains, evaluates, and versions a Random Forest model for wine quality prediction.

## Dataset
- **Source**: Wine Quality Dataset
- **Total Samples**: 6,497 wines
- **Features**: 12 chemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
  - Type (red/white)
- **Target Variable**: Quality (3-9 scale)
- **Number of Classes**: 7

## Model Architecture
### Algorithm
Random Forest Classifier

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| n_estimators | 150 |
| max_depth | 10 |
| random_state | 42 |
| n_jobs | -1 |

## Performance Metrics
| Metric | Training | Testing |
|--------|----------|---------|
| Accuracy | 82.14% | 64.00% |
| F1 Score | 0.81 | 0.62 |

### Classification Report (Test Set)
| Quality | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| 3 | 0.00 | 0.00 | 0.00 | Low |
| 4 | 0.33 | 0.02 | 0.04 | Low |
| 5 | 0.69 | 0.67 | 0.68 | High |
| 6 | 0.61 | 0.77 | 0.68 | High |
| 7 | 0.65 | 0.47 | 0.54 | Medium |
| 8 | 1.00 | 0.18 | 0.30 | Low |
| 9 | 0.00 | 0.00 | 0.00 | Low |

## Project Structure
```
git_lab2/
│
├── .github/
│   └── workflows/
│       ├── model_calibration_on_push.yml  # Triggers on push
│       └── model_calibration.yml          # Daily scheduled run
│
├── src/
│   ├── train_model.py        # Model training pipeline
│   ├── evaluate_model.py     # Model evaluation pipeline
│   └── test.py               # Unit tests
│
├── models/                    # Versioned model storage
│   └── wine_model_*.joblib
│
├── metrics/                   # Performance metrics
│   └── *_metrics.json
│
├── data/                     # Data storage
│   ├── train_data.pickle
│   ├── test_data.pickle
│   └── scaler.pickle
│
├── mlruns/                   # MLflow tracking
│
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

## Technologies Used
- **Programming Language**: Python 3.9
- **Machine Learning**: Scikit-learn 1.3.0
- **Data Processing**: Pandas 2.0.3, NumPy 1.24.3
- **Experiment Tracking**: MLflow 2.7.1
- **Model Serialization**: Joblib 1.3.1
- **CI/CD**: GitHub Actions

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git

### Local Setup
1. Clone the repository:
```bash
   git clone https://github.com/YOUR_USERNAME/wine-quality-mlops.git
   cd wine-quality-mlops
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
   python src/train_model.py --timestamp "local_test"
```

4. Evaluate the model:
```bash
   python src/evaluate_model.py --timestamp "local_test"
```

## GitHub Actions Workflow

### Continuous Integration Pipeline
The pipeline automatically executes upon:
- **Push Events**: Triggered on push to main branch
- **Scheduled Runs**: Daily at midnight UTC

### Workflow Steps
1. **Environment Setup**
   - Configure Python 3.9
   - Install dependencies from requirements.txt

2. **Model Training**
   - Load wine quality dataset
   - Preprocess and scale features
   - Train Random Forest model
   - Track experiments with MLflow

3. **Model Evaluation**
   - Load test data
   - Generate predictions
   - Calculate performance metrics
   - Save classification report

4. **Versioning**
   - Save model with timestamp
   - Store metrics in JSON format
   - Commit changes to repository

## Key Features
- **Automated Training**: Model retraining on every code push
- **Version Control**: Timestamp-based model versioning
- **Experiment Tracking**: MLflow integration for parameter and metric logging
- **Scalable Pipeline**: Handles 6,497 samples efficiently
- **Comprehensive Evaluation**: Detailed classification metrics and reports

## Results and Observations
- The model achieves 82% accuracy on training data and 64% on test data
- Best performance on quality ratings 5 and 6 (most common classes)
- Feature importance analysis shows alcohol content and volatile acidity as top predictors
- Model demonstrates robust performance despite class imbalance

## Future Improvements
- Implement hyperparameter tuning using GridSearchCV
- Address class imbalance with SMOTE or class weights
- Add cross-validation for more robust evaluation
- Implement model monitoring and drift detection
- Create API endpoint for model serving

## Author Information
**Name**: Priyanka Raj Rajendran
**Course**: Machine Learning Operations (MLOps)  
**Institution**: Northeastern University  




