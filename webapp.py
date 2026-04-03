import streamlit as st
import cv2
import numpy as np
import time
import threading
from collections import deque
from hand_tracker import HandTracker
import pytesseract
import easyocr
import base64
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="AIR WRITING 2.0", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZE OCR ENGINES ---
@st.cache_resource
def load_ocr():
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        return reader
    except:
        return None

try:
    easy_reader = load_ocr()
except:
    easy_reader = None

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1c24 !important;
    }
    .status-card {
        padding: 10px;
        border-radius: 10px;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.2);
        margin-bottom: 10px;
    }
    .text-output {
        font-size: 32px;
        font-weight: bold;
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0,255,255,0.7);
        padding: 15px;
        border-left: 4px solid #00ffff;
        background: rgba(0,255,255,0.05);
    }
    .stSlider > div > div > div > div {
        background-color: #00ffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'recognized_history' not in st.session_state:
    st.session_state.recognized_history = []
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""
if 'is_writing' not in st.session_state:
    st.session_state.is_writing = False

# --- UI Layout ---
st.sidebar.title("🛠️ System Control")
cam_id = st.sidebar.number_input("Camera Index", 0, 3, 0)
use_dshow = st.sidebar.checkbox("Windows High-Compatibility (DSHOW)", value=True)
brush_size = st.sidebar.slider("Brush Thickness", 5, 50, 12)
color_picker = st.sidebar.color_picker("Ink Color", "#00FFFF")
eraser_mode = st.sidebar.toggle("Eraser Mode")
video_off = st.sidebar.toggle("Privacy Mode (Video Off)")
auto_recognition = st.sidebar.checkbox("Auto-Recognition", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📜 Transcription History")
for i, txt in enumerate(reversed(st.session_state.recognized_history[-20:])):
    st.sidebar.markdown(f"**{len(st.session_state.recognized_history)-i}.** {txt}")

# Download transcript
if st.session_state.recognized_history:
    transcript = "\n".join(st.session_state.recognized_history)
    st.sidebar.download_button("📥 Download Transcript", transcript, file_name="air_writing_session.txt")

# Main View
st.title("🖊️ AIR WRITING 2.0")
st.markdown("---")

col_cam, col_info = st.columns([3.5, 1.5])

with col_info:
    st.subheader("System Status")
    status_placeholder = st.empty()
    st.markdown("""
    <div class="status-card">
        <b>Gestures Guide:</b><br>
        ☝️ <b>1 Finger Up:</b> Draw Ink<br>
        ✌️ <b>2 Fingers Up:</b> Hover Mode<br>
        🖐️ <b>Open Palm:</b> Clear Canvas
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Current Recognition")
    text_placeholder = st.empty()
    
    if st.button("🗑️ Clear Canvas Now"):
        st.session_state.clear_triggered = True
    
    if st.button("🔄 Hard Reset System"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Camera area with col_cam
with col_cam:
    cam_placeholder = st.empty()

# --- OCR THREAD ---
def process_ocr(canvas_img):
    try:
        # Preprocess
        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        white_bg = 255 - binary
        
        result = ""
        # Try Tesseract
        try:
            result = pytesseract.image_to_string(white_bg, config='--psm 6').strip()
        except:
            pass
            
        # Try EasyOCR fallback
        if not result and easy_reader:
            ocr_input = cv2.cvtColor(white_bg, cv2.COLOR_GRAY2BGR)
            res = easy_reader.readtext(ocr_input, detail=0)
            result = " ".join(res) if res else ""
        
        if result and result not in ["No text detected", "Could not read"]:
            st.session_state.recognized_history.append(result)
            st.session_state.last_text = result
    except Exception as e:
        st.write(f"[DEBUG] OCR ERROR: {e}")

# --- CACHED RESOURCES ---
@st.cache_resource
def get_tracker():
    return HandTracker(max_hands=1, detection_con=0.85, track_con=0.5)

tracker = get_tracker()

# --- MAIN LOOP ---
if 'system_running' not in st.session_state:
    st.session_state.system_running = False

# Start/Stop Button
if not st.session_state.system_running:
    if st.button("🚀 START AIR WRITING SYSTEM", use_container_width=True):
        st.session_state.system_running = True
        st.rerun()

# Converter hex to BGR
hex_color = color_picker.lstrip('#')
rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

if eraser_mode:
    bgr_color = (0, 0, 0)
    brush_size = 50

# Camera initialization
if st.session_state.system_running:
    if 'cap' not in st.session_state or st.session_state.cap is None:
        st.sidebar.info(f"Opening Camera {cam_id}...")
        try:
            # Try specified mode first, then fallback to other
            modes = [cv2.CAP_DSHOW, 0] if use_dshow else [0, cv2.CAP_DSHOW]
            success_conn = False
            for mod in modes:
                dev_cap = cv2.VideoCapture(cam_id, mod)
                if dev_cap.isOpened():
                    ret, _ = dev_cap.read()
                    if ret:
                        st.session_state.cap = dev_cap
                        st.sidebar.success(f"Connected using Mode {mod}! ✅")
                        success_conn = True
                        break
                dev_cap.release()
            
            if not success_conn:
                st.sidebar.error("Hardware denied access. Check privacy settings!")
                st.session_state.system_running = False
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.session_state.system_running = False

    if st.session_state.get('cap'):
        cap = st.session_state.cap
        canvas = None
        xp, yp = 0, 0
        last_writing_time = time.time()
        actively_writing = False
        prev_time = time.time()
        
        # UI Elements
        stop_app = st.sidebar.button("🟥 SHUTDOWN SYSTEM", key="shutdown_btn")
        fps_placeholder = st.sidebar.empty()

        if stop_app:
            st.session_state.system_running = False
            cap.release()
            st.session_state.cap = None
            st.rerun()

        # Kalman Filter
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.03
        
        def kf_predict(x, y):
            measured = np.array([[np.float32(x)], [np.float32(y)]])
            kf.correct(measured)
            pred = kf.predict()
            return int(pred[0]), int(pred[1])

        # Main Real-time Loop
        while st.session_state.system_running:
            success, img = cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            
            # FPS
            curr_time = time.time()
            fps = 1 / (max(0.001, curr_time - prev_time))
            prev_time = curr_time
            fps_placeholder.markdown(f"**Performance:** `{int(fps)} FPS`")
            
            if canvas is None:
                canvas = np.zeros_like(img)
            
            # External Clear
            if st.session_state.get('clear_triggered', False):
                canvas = np.zeros_like(img)
                st.session_state.clear_triggered = False

            img = tracker.find_hands(img, draw=False)
            lm_list = tracker.find_position(img, draw=False)
            
            current_status = "IDLE ⚪"
            currently_writing = False

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]
                fingers = tracker.fingers_up()
                
                if fingers[1] and not fingers[2] and not fingers[3]:
                    if not actively_writing:
                        kf.statePre = np.array([[x1], [y1], [0], [0]], np.float32)
                        kf.statePost = np.array([[x1], [y1], [0], [0]], np.float32)
                    
                    currently_writing = True
                    actively_writing = True
                    last_writing_time = time.time()
                    current_status = "WRITING 🖋️"
                    
                    sx, sy = kf_predict(x1, y1)
                    dot_col = (255, 255, 255) if eraser_mode else bgr_color
                    cv2.circle(img, (sx, sy), brush_size // 2 + 2, dot_col, cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = sx, sy
                    
                    cv2.line(canvas, (xp, yp), (sx, sy), bgr_color, brush_size, cv2.LINE_AA)
                    xp, yp = sx, sy
                elif fingers[1] and fingers[2]:
                    xp, yp = 0, 0
                    current_status = "HOVERING 🔍"
                elif all(f == 1 for f in fingers):
                    canvas = np.zeros_like(img)
                    xp, yp = 0, 0
                    current_status = "CLEARING 🧹"
                else:
                    xp, yp = 0, 0
                    
            if actively_writing and not currently_writing:
                elapsed = time.time() - last_writing_time
                if auto_recognition and elapsed > 2.5:
                    threading.Thread(target=process_ocr, args=(canvas.copy(),), daemon=True).start()
                    actively_writing = False

            # Display
            if video_off:
                cam_placeholder.image(canvas if np.any(canvas) else np.zeros_like(img), channels="BGR", use_container_width=True)
            else:
                gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, inv_canvas = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
                inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
                merged = cv2.bitwise_and(img, inv_canvas)
                merged = cv2.bitwise_or(merged, canvas)
                cam_placeholder.image(merged, channels="BGR", use_container_width=True)
            
            status_placeholder.markdown(f"**Status:** `{current_status}`")
            text_placeholder.markdown(f'<div class="text-output">{st.session_state.last_text if st.session_state.last_text else "..."}</div>', unsafe_allow_html=True)
            
            time.sleep(0.001)
    else:
        st.error("Please connect a camera or grant permissions.")
