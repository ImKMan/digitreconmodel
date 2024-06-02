from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model("digit_recognition_model.h5")


@app.route("/")
def home():
    return "Digit Recognition Model"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data["data"]).reshape(1, 32, 32, 1)
    prediction = model.predict(input_data)
    result = np.argmax(prediction, axis=1)[0]
    return jsonify({"prediction": int(result)})


if __name__ == "__main__":
    app.run(debug=True)
