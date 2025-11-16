import pickle, os, json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import joblib
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    
    print("="*50)
    print("EVALUATING WINE QUALITY MODEL")
    print("="*50)
    
    try:
        model_version = f'wine_model_{timestamp}_rf_model'
        model = joblib.load(f'{model_version}.joblib')
        print(f"Model loaded: {model_version}")
    except:
        raise ValueError('Failed to load the latest model')
    
    try:
        # Load test data
        with open('data/test_data.pickle', 'rb') as f:
            X_test, y_test = pickle.load(f)
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except:
        raise ValueError("Test data not found. Please run train_model.py first.")
    
    # Make predictions
    y_predict = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, average='weighted')
    
    print(f"\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Get unique classes for the report
    unique_classes = sorted(set(y_test))
    print(f"Number of quality classes: {len(unique_classes)}")
    print(f"Quality classes: {unique_classes}")
    
    # Classification report
    report = classification_report(y_test, y_predict, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    
    print("\nClassification Report Summary:")
    for class_label in unique_classes:
        class_str = str(class_label)
        if class_str in report:
            print(f"Quality {class_label}: "
                  f"precision={report[class_str]['precision']:.3f}, "
                  f"recall={report[class_str]['recall']:.3f}, "
                  f"f1-score={report[class_str]['f1-score']:.3f}")
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "model_version": model_version,
        "test_samples": len(y_test),
        "unique_quality_scores": [int(x) for x in unique_classes]
    }
    
    # Save metrics
    if not os.path.exists('metrics/'): 
        os.makedirs("metrics/")
    
    metrics_filename = f'{timestamp}_metrics.json'
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    print(f"\nMetrics saved to: {metrics_filename}")
    print("="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)