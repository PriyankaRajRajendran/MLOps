import pandas as pd
import numpy as np
import mlflow, datetime, os, pickle
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    print("="*50)
    print("TRAINING WINE QUALITY MODEL")
    print("="*50)
    
    # Try multiple locations for the dataset
    dataset_paths = [
        "/Users/priyankaraj/Downloads/wine_quality_merged (1).csv",  # Your local path
        "./data/wine_quality.csv",  # Project folder path
        "wine_quality.csv",  # Root path
        "../data/wine_quality.csv"  # Alternative path
    ]
    
    df = None
    for path in dataset_paths:
        try:
            df = pd.read_csv(path)
            print(f"✅ Dataset loaded successfully from: {path}")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            break
        except:
            continue
    
    # If CSV not found, use sklearn wine dataset as fallback
    if df is None:
        print("⚠️ CSV not found in any location, using sklearn wine dataset as fallback...")
        from sklearn.datasets import load_wine
        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = pd.Series(wine.target)
        print(f"Sklearn wine dataset loaded: {X.shape}")
    else:
        # Process your CSV data
        print("\nProcessing CSV data...")
        print(f"Dataset info:")
        print(df.info())
        
        # Show first few rows
        print("\nFirst 3 rows of dataset:")
        print(df.head(3))
        
        # Handle the 'type' column if it exists (convert to numeric)
        if 'type' in df.columns:
            print("\n✅ Found 'type' column - converting to numeric (red=0, white=1)")
            df['type'] = df['type'].map({'red': 0, 'white': 1})
            print(f"Type values converted: red→0, white→1")
        
        # Determine target column
        # Common wine quality dataset column names
        possible_target_columns = ['quality', 'Quality', 'target', 'class', 'label']
        
        target_column = None
        for col in possible_target_columns:
            if col in df.columns:
                target_column = col
                break
        
        if target_column:
            print(f"\n✅ Found target column: '{target_column}'")
            X = df.drop(target_column, axis=1)
            y = df[target_column]
        else:
            # If no standard target column found, use last column
            print(f"\n⚠️ No standard target column found, using last column: '{df.columns[-1]}'")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Handle any missing values
        if X.isnull().any().any():
            print("\n⚠️ Missing values detected, filling with median...")
            X = X.fillna(X.median())
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Feature columns: {X.columns.tolist()}")
        print(f"Target shape: {y.shape}")
        print(f"Target unique values: {sorted(y.unique())}")
        print(f"Target value counts:\n{y.value_counts().sort_index()}")
    
    # Convert to numpy arrays if needed
    X = X.values if hasattr(X, 'values') else X
    y = y.values if hasattr(y, 'values') else y
    
    # Split data
    print("\n" + "="*30)
    print("Splitting data into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except:
        # If stratify fails (too few samples), try without it
        print("⚠️ Stratified split failed, using regular split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Scale features (important for wine quality)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled successfully")
    
    # Save data for evaluation
    if not os.path.exists('data'): 
        os.makedirs('data/')
        print("✅ Created data/ directory")
    
    with open('data/train_data.pickle', 'wb') as f:
        pickle.dump((X_train_scaled, y_train), f)
    
    with open('data/test_data.pickle', 'wb') as f:
        pickle.dump((X_test_scaled, y_test), f)
    
    with open('data/scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Data saved for evaluation")
    
    # MLflow tracking
    print("\n" + "="*30)
    print("Setting up MLflow tracking...")
    
    if not os.path.exists('mlruns'):
        os.makedirs('mlruns/')
        print("✅ Created mlruns/ directory")
    
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Wine_Quality_Dataset"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    
    try:
        experiment_id = mlflow.create_experiment(f"{experiment_name}")
    except:
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(f"{experiment_name}")
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            # Create with a unique name
            experiment_name = f"{dataset_name}_{current_time}_{np.random.randint(1000)}"
            experiment_id = mlflow.create_experiment(f"{experiment_name}")
    
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_name}"):
        
        # Log parameters
        params = {
            "dataset_name": dataset_name,
            "model_type": "RandomForest",
            "n_estimators": 150,
            "max_depth": 10,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(len(set(y)))
        }
        
        mlflow.log_params(params)
        
        # Train model
        print("\n" + "="*30)
        print("Training Random Forest model...")
        print("Parameters: n_estimators=150, max_depth=10")
        
        forest = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        forest.fit(X_train_scaled, y_train)
        print("✅ Model training complete!")
        
        # Evaluate on training data
        print("\nEvaluating model performance...")
        y_train_pred = forest.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        
        # Evaluate on test data
        y_test_pred = forest.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"\n📊 Model Performance:")
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Training F1 Score: {train_f1:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'train_accuracy': float(train_acc),
            'train_f1_score': float(train_f1),
            'test_accuracy': float(test_acc),
            'test_f1_score': float(test_f1)
        })
        
        # Create models directory
        if not os.path.exists('models/'): 
            os.makedirs("models/")
            print("\n✅ Created models/ directory")
        
        # Save model with versioning
        model_version = f'wine_model_{timestamp}'
        model_filename = f'{model_version}_rf_model.joblib'
        dump(forest, model_filename)
        
        print(f"\n✅ Model saved as: {model_filename}")
        
        # Feature importance (if we have feature names)
        print("\n📊 Feature Importance (Top 5):")
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X.shape[1])],
            'importance': forest.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")
        
        print("\n" + "="*50)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*50)
        print(f"\nSummary:")
        print(f"  - Model: Random Forest Classifier")
        print(f"  - Dataset: {X_train.shape[0] + X_test.shape[0]} samples")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Classes: {len(set(y))}")
        print(f"  - Test Accuracy: {test_acc:.2%}")
        print(f"  - Model saved: {model_filename}")
        print(f"  - Metrics logged to MLflow")
        print("="*50)