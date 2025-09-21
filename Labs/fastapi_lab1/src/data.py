# src/data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_clean_data(csv_path: str = "/Users/priyankaraj/Documents/wine_quality_merged.csv"):
    df = pd.read_csv(csv_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Rename columns: replace spaces with underscores and lowercase
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]

    # Encode any categorical columns
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["quality"])
    y = df["quality"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_clean_data()
    print(f"Data loaded and split. Training size: {len(X_train)}, Test size: {len(X_test)}")
