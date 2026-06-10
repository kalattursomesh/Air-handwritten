from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json

app = Flask(__name__)
DATA_DIR = "training_data"

def get_label_map():
    map_file = os.path.join(DATA_DIR, "label_map.json")
    if os.path.exists(map_file):
        try:
            with open(map_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_label_map(label_map):
    os.makedirs(DATA_DIR, exist_ok=True)
    map_file = os.path.join(DATA_DIR, "label_map.json")
    try:
        with open(map_file, "w") as f:
            json.dump(label_map, f, indent=2)
    except Exception as e:
        print(f"Error saving label map: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/load_dataset")
def load_dataset():
    """Load all saved drawings and return them as JSON for frontend TF.js."""
    samples = []
    label_map = get_label_map()
    
    for label_name in label_map.keys():
        label_dir = os.path.join(DATA_DIR, label_name)
        if os.path.exists(label_dir):
            for fname in os.listdir(label_dir):
                if fname.endswith(".npy"):
                    filepath = os.path.join(label_dir, fname)
                    try:
                        img = np.load(filepath)
                        # Normalize to [0, 1] for TF.js
                        pixels = (img.astype(float) / 255.0).flatten().tolist()
                        samples.append({
                            "label": label_name,
                            "pixels": pixels
                        })
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        
    return jsonify({
        "samples": samples,
        "label_map": label_map
    })

@app.route("/api/save", methods=["POST"])
def save_sample():
    """Save a drawing with its label (called locally)."""
    data = request.json
    label = data.get("label", "").strip().upper()
    pixels = data.get("pixels", [])
    
    if not label or not pixels or len(pixels) != 784:
        return jsonify({"error": "Missing or invalid label/pixels data"}), 400
        
    label_map = get_label_map()
    if label not in label_map:
        label_map[label] = len(label_map)
        save_label_map(label_map)
        
    # Save sample image
    label_dir = os.path.join(DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    
    count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    filepath = os.path.join(label_dir, f"sample_{count}.npy")
    
    try:
        # Save as 28x28 grayscale image
        img_array = np.array(pixels, dtype=np.uint8).reshape(28, 28)
        np.save(filepath, img_array)
        return jsonify({"success": True, "label": label, "count": count + 1})
    except Exception as e:
        return jsonify({"error": f"Failed to save sample: {str(e)}"}), 500

@app.route("/api/stats")
def get_stats():
    """Get training data statistics."""
    label_map = get_label_map()
    stats = {}
    total = 0
    for label in label_map.keys():
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.exists(label_dir):
            count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
            stats[label] = count
            total += count
            
    return jsonify({
        "stats": stats,
        "total": total,
        "classes": len(stats),
        "device": "cpu",
        "gpu": None,
        "model_loaded": False
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AIR WRITING AI — Neural Handwriting Recognition (Vercel Mode)")
    print("="*50)
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)
