 from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load("random_forest_model_predictp.joblib")
scaler = joblib.load("random_forest_scaler_predictp.joblib")

@app.route('/')
def home():
    return "Seawater Pollution Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON input
    data = request.get_json(force=True)
    # Convert the input to an array and reshape
    input_features = np.array(data['input']).reshape(1, -1)
    # Scale the input if using scaler
    input_scaled = scaler.transform(input_features)
    # Predict with the model
    prediction = model.predict(input_scaled)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

