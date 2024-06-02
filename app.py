from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import requests

app = Flask(__name__)


# Function to download model
def download_model():
    url = "https://your-storage-service.com/path/to/digit_recognition_model.h5"  # Replace with your model URL
    response = requests.get(url)
    with open("digit_recognition_model.h5", "wb") as f:
        f.write(response.content)


# Check if model exists, if not, download it
if not os.path.exists("digit_recognition_model.h5"):
    download_model()

# Load the model
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
        try:
            image = Image.open(file).convert("L")  # Convert to grayscale
            image = image.resize((32, 32))  # Resize to 32x32
            image = img_to_array(image)  # Convert to array
            image = image.reshape(1, 32, 32, 1)  # Reshape to fit model input
            image = image.astype("float32") / 255  # Normalize pixel values
            prediction = model.predict(image)
            result = np.argmax(prediction, axis=1)[0]
            return jsonify({"prediction": int(result)})
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
