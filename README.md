# Air Writing AI ✍️

A high-performance, browser-native Air Writing and Gesture Recognition system built with **MediaPipe Hands**, **Flask**, and **PyTorch**. 

Draw characters in the air using your webcam, train your own custom handwriting model, and let the AI recognize your gestures in real-time.

## Features ✨
- **Zero-Lag MediaPipe Tracking**: Camera and hand tracking run completely in the browser using JavaScript to eliminate latency.
- **Trainable CNN Model (PyTorch)**: Teach the AI to recognize your specific handwriting style.
- **Live Prediction Mode**: Real-time handwriting recognition with confidence scoring.
- **Cyber-Teal Premium UI**: Beautiful, interactive web interface.

## Tech Stack 🛠️
- **Frontend**: HTML5 Canvas, Vanilla JS, CSS (Glassmorphism design)
- **Computer Vision**: [MediaPipe Hands](https://mediapipe.dev/) (JS version)
- **Backend**: Flask
- **Machine Learning**: PyTorch (Custom 3-Layer Convolutional Neural Network)

## Installation & Setup 🚀

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kalattursomesh/Air-handwritten.git
   cd Air-handwritten
   ```

2. **Install dependencies:**
   Make sure you have Python installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```
4. **Open in Browser:** Navigate to `http://localhost:5000`

## How to Use 📖

### 1. Training Mode
- Select **Train Mode**.
- Lift your **index finger** to the camera.
- Move your finger to draw a character on the screen.
- Enter the label of the character you drew (e.g., `A`) and click **Save**.
- Repeat this for 5-10 samples per character.
- Click **Train Model** to generate your custom PyTorch model.

### 2. Predict Mode
- Switch to **Predict Mode**.
- Draw a character in the air.
- Click **Recognize** to see the CNN's prediction.

### Gestures:
- ☝️ **Index finger only** → Draw ink
- ✌️ **Two fingers** → Pause / Hover
- 🖐️ **Open palm** → Clear canvas
- ✊ **Fist** → Stop stroke

## Architecture Details
This app completely offloads the heavy Computer Vision (MediaPipe) processing to the client-side browser, ensuring a smooth, real-time 60 FPS drawing experience without overloading the Python backend. The backend strictly serves as the PyTorch AI brain for image classification.
