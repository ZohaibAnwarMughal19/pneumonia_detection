# ============================================
# Pneumonia Detection - predict.py
# Phase 3: Image predict karna
# ============================================

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import json

# ============================================
# STEP 1: MODEL LOAD KARO
# ============================================
MODEL_PATH = r"E:\python\ML\pnumonia\model\pneumonia_model_v2.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# ============================================
# STEP 2: IMAGE PATH ARGUMENT SE LO
# Command line se image path milega
# Usage: py -3.11 predict.py "image_path.jpg"
# ============================================
if len(sys.argv) < 2:
    print(json.dumps({
        "error": "Koi image path nahi diya!"
    }))
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# ============================================
# STEP 3: IMAGE PREPROCESS KARO
# Bilkul waise jaise training mein kiya tha
# ============================================
try:
    img = Image.open(IMAGE_PATH).convert('RGB')  # RGB mein convert karo
    img = img.resize((150, 150))                  # 150x150 resize karo
    img_array = np.array(img) / 255.0             # normalize karo
    img_array = np.expand_dims(img_array, axis=0) # batch dimension add karo

except Exception as e:
    print(json.dumps({
        "error": f"Image load nahi hui: {str(e)}"
    }))
    sys.exit(1)

# ============================================
# STEP 4: PREDICTION KARO
# ============================================
prediction = model.predict(img_array, verbose=0)
confidence = float(prediction[0][0])

# ============================================
# STEP 5: RESULT DECIDE KARO
# 0.5 se upar = PNEUMONIA
# 0.5 se neeche = NORMAL
# ============================================
if confidence >= 0.5:
    result    = "PNEUMONIA"
    percent   = round(confidence * 100, 2)
    message   = "X-ray mein Pneumonia ke signs hain!"
else:
    result    = "NORMAL"
    percent   = round((1 - confidence) * 100, 2)
    message   = "X-ray Normal hai, koi Pneumonia nahi!"

# ============================================
# STEP 6: JSON FORMAT MEIN RESULT DO
# PHP baad mein yeh JSON read karega
# ============================================
output = {
    "result":     result,
    "confidence": percent,
    "message":    message
}

print(json.dumps(output))