# src/train.py

from sklearn.linear_model import LinearRegression
from joblib import dump
from pathlib import Path
from src.data import load_clean_data

def train_model(model_path: str = "model/wine_model.pkl"):
    X_train, X_test, y_train, y_test = load_clean_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()
