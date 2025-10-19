import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Loading Wine dataset...")
# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class names: {wine.target_names}")
print(f"Feature names: {wine.feature_names}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Building model...")
# Build neural network
model = keras.Sequential([
    layers.Input(shape=(13,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Train
print("Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Save model and preprocessing objects
print("Saving model and preprocessing objects...")
model.save('wine_model.keras')

import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names and target names
with open('wine_info.pkl', 'wb') as f:
    pickle.dump({
        'feature_names': wine.feature_names,
        'target_names': wine.target_names
    }, f)

print("Model saved successfully!")
print(f"Classes: {list(wine.target_names)}")