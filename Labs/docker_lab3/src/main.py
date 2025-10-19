from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle

app = Flask(__name__, static_folder='statics')

# Load the trained model
print("Loading model...")
model = keras.models.load_model('wine_model.keras')
print("Model loaded successfully!")

# Load scaler and wine info
print("Loading preprocessing objects...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('wine_info.pkl', 'rb') as f:
    wine_info = pickle.load(f)

feature_names = wine_info['feature_names']
target_names = wine_info['target_names']

print(f"Classes: {target_names}")
print("Ready to make predictions!")

@app.route('/')
def home():
    return "Welcome to the Wine Quality Classification API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data - all 13 features
            features = []
            features.append(float(request.form['alcohol']))
            features.append(float(request.form['malic_acid']))
            features.append(float(request.form['ash']))
            features.append(float(request.form['alcalinity']))
            features.append(float(request.form['magnesium']))
            features.append(float(request.form['total_phenols']))
            features.append(float(request.form['flavanoids']))
            features.append(float(request.form['nonflavanoid_phenols']))
            features.append(float(request.form['proanthocyanins']))
            features.append(float(request.form['color_intensity']))
            features.append(float(request.form['hue']))
            features.append(float(request.form['od280']))
            features.append(float(request.form['proline']))
            
            # Prepare input array
            input_data = np.array([features])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled, verbose=0)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
            
            # Get wine class name
            wine_class = target_names[predicted_class]
            
            return jsonify({
                "wine_class": wine_class,
                "confidence": confidence,
                "probabilities": {
                    target_names[0]: float(prediction[0][0]),
                    target_names[1]: float(prediction[0][1]),
                    target_names[2]: float(prediction[0][2])
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    else:
        # GET request - return the HTML form
        return render_template('predict.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)