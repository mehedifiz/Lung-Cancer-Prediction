from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("lung_cancer_logistic_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        data['AGE'],
        data['SMOKING'],
        data['YELLOW_FINGERS'],
        data['ANXIETY'],
        data['PEER_PRESSURE'],
        data['CHRONIC_DISEASE'],
        data['FATIGUE'],
        data['ALLERGY'],
        data['WHEEZING'],
        data['ALCOHOL_CONSUMING'],
        data['COUGHING'],
        data['SHORTNESS_OF_BREATH'],
        data['SWALLOWING_DIFFICULTY'],
        data['CHEST_PAIN']
    ]

    prediction = model.predict([features])[0]
    return jsonify({'prediction': str(prediction)})
