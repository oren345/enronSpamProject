from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from text_utils import preprocess_text
import numpy as np

app = Flask(__name__)
CORS(app)

# =============================
# MODEL YÜKLE
# =============================
model = load_model("../model/enron_dnn_bilstm.keras")

# =============================
# PREDICT ENDPOINT
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    # Eğitimde kullanılan preprocessing ile AYNI
    text = preprocess_text(data["text"])

    # Modele string tensor ver
    prob = float(
        model.predict(
            np.array([text], dtype=object),
            verbose=0
        )[0][0]
    )

    # =============================
    # THRESHOLD TUNING
    # =============================
    if prob >= 0.7:
        label = "Spam"
    elif prob <= 0.4:
        label = "Ham"
    else:
        label = "Uncertain"

    return jsonify({
        "label": label,
        "probability": round(prob, 4)
    })

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    app.run(debug=True)

    app.run(debug=True)
