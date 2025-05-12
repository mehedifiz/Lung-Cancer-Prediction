from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model (make sure the file is at the same level as this script)
model = joblib.load("lung_cancer_model.pkl")

@app.route("/")
def home():
    return "✅ Lung Cancer Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            data.get("age", 0),
            data.get("smoking", 0),
            data.get("yellow_fingers", 0),
            data.get("anxiety", 0),
            data.get("peer_pressure", 0),
            data.get("chronic_disease", 0),
            data.get("fatigue", 0),
            data.get("allergy", 0),
            data.get("wheezing", 0),
            data.get("alcohol_consumption", 0),
            data.get("coughing", 0),
            data.get("shortness_of_breath", 0),
            data.get("swallowing_difficulty", 0),
            data.get("chest_pain", 0)
        ]
        prediction = model.predict(np.array([features]))[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# THIS IS CRUCIAL FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
