from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("lung_cancer_model.pkl")

@app.route("/")
def home():
    return "Lung Cancer Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        age = data.get("age")
        smoking = data.get("smoking")
        yellow_fingers = data.get("yellow_fingers")
        anxiety = data.get("anxiety")
        peer_pressure = data.get("peer_pressure")
        chronic_disease = data.get("chronic_disease")
        fatigue = data.get("fatigue")
        allergy = data.get("allergy")
        wheezing = data.get("wheezing")
        alcohol_consumption = data.get("alcohol_consumption")
        coughing = data.get("coughing")
        shortness_of_breath = data.get("shortness_of_breath")
        swallowing_difficulty = data.get("swallowing_difficulty")
        chest_pain = data.get("chest_pain")

        features = np.array([[age, smoking, yellow_fingers, anxiety, peer_pressure,
                              chronic_disease, fatigue, allergy, wheezing,
                              alcohol_consumption, coughing, shortness_of_breath,
                              swallowing_difficulty, chest_pain]])

        prediction = model.predict(features)[0]
        result = {"prediction": int(prediction)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Bind to the port Render provides
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
