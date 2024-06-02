from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("digit_recognition_model.h5")


def preprocess_image(image):
    image = image.resize((32, 32))
    image = image.convert("L")
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 32, 32, 1)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_digit = np.argmax(predictions, axis=1)[0]
        response = {"prediction": int(predicted_digit)}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
