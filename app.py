from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from model import ModelTrainer

app = Flask(__name__)
trainer = ModelTrainer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/save", methods=["POST"])
def save_sample():
    """Save a drawing with its label."""
    data = request.json
    label = data.get("label", "").strip().upper()
    image_data = data.get("image", "")
    
    if not label or not image_data:
        return jsonify({"error": "Missing label or image"}), 400
    
    # Decode base64 canvas image
    img_bytes = base64.b64decode(image_data.split(",")[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    count = trainer.save_sample(img_resized, label)
    return jsonify({"success": True, "label": label, "count": count})

@app.route("/api/train", methods=["POST"])
def train_model():
    """Train the CNN on collected samples."""
    result = trainer.train(epochs=30)
    return jsonify(result)

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict a character from drawing."""
    data = request.json
    image_data = data.get("image", "")
    
    if not image_data:
        return jsonify({"error": "Missing image"}), 400
    
    img_bytes = base64.b64decode(image_data.split(",")[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    result = trainer.predict(img_resized)
    return jsonify(result)

@app.route("/api/stats")
def get_stats():
    """Get training data statistics."""
    stats = trainer.get_stats()
    total = sum(stats.values())
    return jsonify({"stats": stats, "total": total, "classes": len(stats)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
