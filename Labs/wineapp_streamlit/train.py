import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

def train_wine_model():
    """Train a model to predict wine quality"""
    
    print("Loading wine quality dataset...")
    
    # UPDATE THIS PATH TO WHERE YOUR CSV FILE IS LOCATED
    # Examples:
    # Windows: data_path = r"C:\Users\YourName\Downloads\wine_quality_merged.csv"
    # Mac/Linux: data_path = "/Users/YourName/Downloads/wine_quality_merged.csv"
    data_path = "/Users/priyankaraj/Downloads/wine_quality_merged.csv"  # UPDATE THIS!
    
    try:
        df = pd.read_csv(data_path)
        print(f"Dataset loaded: {len(df)} samples")
    except FileNotFoundError:
        print(f"Dataset not found at {data_path}")
        print("Please update the data_path variable with the correct location of your CSV file")
        return
    
    # Display dataset info
    print("\nDataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("\nQuality distribution:")
    print(df['quality'].value_counts().sort_index())
    
    # Create binary classification: high quality (>=7) vs standard quality (<7)
    df['high_quality'] = (df['quality'] >= 7).astype(int)
    
    print(f"\nHigh quality wines: {df['high_quality'].sum()} ({df['high_quality'].mean():.2%})")
    print(f"Standard quality wines: {(1-df['high_quality']).sum()} ({(1-df['high_quality']).mean():.2%})")
    
    # Prepare features and target
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 
        'residual sugar', 'chlorides', 'free sulfur dioxide',
        'total sulfur dioxide', 'density', 'pH', 'sulphates', 
        'alcohol', 'type'
    ]
    
    X = df[feature_columns].copy()
    y = df['high_quality']
    
    # Encode wine type (red/white) if it's string
    if X['type'].dtype == 'object':
        le = LabelEncoder()
        X['type'] = le.fit_transform(X['type'])
        print(f"\nEncoded wine types: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Standard Quality', 'High Quality']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head().to_string(index=False))
    
    # Save the model
    model_path = 'wine_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    print("\n✅ Training complete! You can now start the FastAPI server and Streamlit app.")

if __name__ == "__main__":
    train_wine_model()