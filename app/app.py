from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("../model/crop_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Crop Recommendation API is running! Use POST /predict with JSON data."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "recommended_crop": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)