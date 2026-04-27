# ============================================
# Pneumonia Detection - app.py
# Flask API for Railway.app deployment
# ============================================

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

# ============================================
# Model load karo
# ============================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'pneumonia_model_v2.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ============================================
# Home route
# ============================================
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "Pneumonia Detection API is live!"
    })

# ============================================
# Predict route
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    # Image check karo
    if 'xray' not in request.files:
        return jsonify({"error": "Koi image nahi mili!"})

    file = request.files['xray']

    try:
        # Image process karo
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict karo
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])

        # Result decide karo
        if confidence >= 0.5:
            result  = "PNEUMONIA"
            percent = round(confidence * 100, 2)
            message = "X-ray mein Pneumonia ke signs hain!"
        else:
            result  = "NORMAL"
            percent = round((1 - confidence) * 100, 2)
            message = "X-ray Normal hai, koi Pneumonia nahi!"

        return jsonify({
            "result":     result,
            "confidence": percent,
            "message":    message
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ============================================
# Run karo
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)