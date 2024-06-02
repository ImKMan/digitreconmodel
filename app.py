from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io

app = Flask(__name__)

model = load_model("digit_recognition_model.h5")


@app.route("/")
def home():
    return "Digit Recognition Model"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file:
        image = Image.open(file).convert("L")
        image = image.resize((32, 32))
        image = img_to_array(image)
        image = image.reshape(1, 32, 32, 1)
        image = image.astype("float32") / 255
        prediction = model.predict(image)
        result = np.argmax(prediction, axis=1)[0]
        return jsonify({"prediction": int(result)})


if __name__ == "__main__":
    app.run(debug=True)
