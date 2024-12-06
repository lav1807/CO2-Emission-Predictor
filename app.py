from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')

# Get the absolute path to the model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values
    engine_size = float(request.json['engine_size'])
    cylinders = int(request.json['cylinders'])
    
    # Make prediction
    prediction = model.predict(np.array([[engine_size, cylinders]]))[0]
    
    # Return prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
