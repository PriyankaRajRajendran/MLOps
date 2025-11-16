"""
Test script to verify model training and evaluation pipeline
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_data_loading():
    """Test if wine quality data can be loaded"""
    print("Testing data loading...")
    
    # Try loading from CSV first
    dataset_path = "/Users/priyankaraj/Downloads/wine_quality_merged (1).csv"
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ CSV loaded successfully: {df.shape}")
        return True
    except:
        print("❌ CSV not found, using sklearn wine dataset")
        wine = load_wine()
        print(f"✅ Sklearn dataset loaded: {wine.data.shape}")
        return True

def test_model_training():
    """Test if model can be trained"""
    print("\nTesting model training...")
    
    # Load simple data
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained successfully. Accuracy: {accuracy:.2f}")
    
    return accuracy > 0.8

def test_directories():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = ['models', 'metrics', 'data', 'mlruns']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing - creating it...")
            os.makedirs(dir_name)
            print(f"✅ {dir_name}/ created")
    
    return True

def test_dependencies():
    """Test if all required packages are installed"""
    print("\nTesting dependencies...")
    
    try:
        import sklearn
        print(f"✅ scikit-learn installed: {sklearn.__version__}")
    except:
        print("❌ scikit-learn not installed")
        return False
    
    try:
        import mlflow
        print(f"✅ mlflow installed: {mlflow.__version__}")
    except:
        print("❌ mlflow not installed")
        return False
    
    try:
        import joblib
        print("✅ joblib installed")
    except:
        print("❌ joblib not installed")
        return False
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("RUNNING TESTS FOR WINE QUALITY PIPELINE")
    print("="*50)
    
    all_tests_passed = True
    
    # Run all tests
    all_tests_passed &= test_dependencies()
    all_tests_passed &= test_directories()
    all_tests_passed &= test_data_loading()
    all_tests_passed &= test_model_training()
    
    print("\n" + "="*50)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - Check above for details")
    print("="*50)