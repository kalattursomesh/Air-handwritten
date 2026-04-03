import cv2
import numpy as np
import time
import threading
from collections import deque
from hand_tracker import HandTracker

# ============================================================
# OCR ENGINES (optional — graceful fallback)
# ============================================================
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    EASYOCR_AVAILABLE = True
    print("[OK] EasyOCR loaded.")
except ImportError:
    print("[WARN] EasyOCR not installed.")

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    print("[OK] Tesseract loaded.")
except Exception:
    print("[WARN] Tesseract not found.")

OCR_AVAILABLE = EASYOCR_AVAILABLE or TESSERACT_AVAILABLE

# ============================================================
# CONFIGURATION
# ============================================================
W_CAM, H_CAM = 1280, 720
BRUSH_THICKNESS = 12
TIMEOUT_DURATION = 2.5
SMOOTHING_WINDOW = 5

# ============================================================
# COLOR PALETTE (BGR)
# ============================================================
PALETTE = [
    {"name": "Teal",   "color": (200, 255, 0),  "thickness": 12},
    {"name": "Green",  "color": (0, 220, 100),  "thickness": 12},
    {"name": "Pink",   "color": (180, 50, 255), "thickness": 12},
    {"name": "Gold",   "color": (50, 230, 255), "thickness": 12},
    {"name": "Red",    "color": (60, 60, 255),  "thickness": 12},
    {"name": "ERASER", "color": (0, 0, 0),      "thickness": 40},
]

TOOLBAR_HEIGHT = 70
SWATCH_SIZE = 40
SWATCH_MARGIN = 15
FOOTER_HEIGHT = 80

# ============================================================
# STATE
# ============================================================
active_color_idx = 0
draw_color = PALETTE[0]["color"]
brush_thickness = PALETTE[0]["thickness"]

xp, yp = 0, 0
canvas = None
last_writing_time = time.time()
is_actively_writing = False
is_drawing_stroke = False
undo_stack = []
recognized_text = ""
recognized_history = []
smoothing_buffer_x = deque(maxlen=SMOOTHING_WINDOW)
smoothing_buffer_y = deque(maxlen=SMOOTHING_WINDOW)
interaction_last_time = 0
is_recognizing = False

# ============================================================
# CAMERA
# ============================================================
def get_working_camera():
    """Try multiple camera indices to find a working one."""
    import sys
    print("[INFO] Searching for camera...")
    for idx in [0, 1, 2, 3]:
        try:
            if "win" in sys.platform:
                test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                test = cv2.VideoCapture(idx)
            if test.isOpened():
                ret, frame = test.read()
                if ret and frame is not None:
                    print(f"[OK] Camera {idx} working.")
                    return test
                test.release()
        except Exception:
            continue
    return None

cap = get_working_camera()
if cap is None:
    print("[ERROR] No camera found! Close other apps using the camera and try again.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)

# ============================================================
# HAND TRACKER
# ============================================================
tracker = HandTracker(max_hands=1, detection_con=0.85, track_con=0.5)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def smooth_point(x, y):
    """Moving average smoothing."""
    smoothing_buffer_x.append(x)
    smoothing_buffer_y.append(y)
    return int(np.mean(smoothing_buffer_x)), int(np.mean(smoothing_buffer_y))

def preprocess_for_ocr(canvas_img):
    """Convert canvas to clean black-on-white for OCR."""
    gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    return 255 - binary

def perform_recognition_thread(canvas_img):
    """Background OCR thread."""
    global recognized_text, is_recognizing
    is_recognizing = True
    try:
        processed = preprocess_for_ocr(canvas_img)
        result = ""

        if TESSERACT_AVAILABLE:
            result = pytesseract.image_to_string(processed, config='--psm 6').strip()

        if not result and EASYOCR_AVAILABLE:
            bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            results = reader.readtext(bgr, detail=1)
            if results:
                texts = [r[1] for r in results if r[2] > 0.2]
                result = " ".join(texts)

        if result and result not in ["No text detected", "Could not read", "Canvas is empty"]:
            recognized_text = result
            recognized_history.append(result)
            print(f"[RECOGNIZED] {result}")
        else:
            recognized_text = "No text detected"
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        recognized_text = "Recognition error"
    is_recognizing = False

def perform_recognition(canvas_img):
    """Trigger OCR in background thread."""
    global recognized_text
    if not OCR_AVAILABLE:
        recognized_text = "[No OCR engine]"
        return
    if np.all(canvas_img == 0):
        recognized_text = "Canvas is empty"
        return
    if not is_recognizing:
        threading.Thread(target=perform_recognition_thread, args=(canvas_img.copy(),), daemon=True).start()
        recognized_text = "Recognizing..."

def save_session():
    """Save transcription history to file."""
    if not recognized_history:
        print("[SAVE] Nothing to save.")
        return
    filename = f"transcription_{int(time.time())}.txt"
    with open(filename, "w") as f:
        f.write("=== AIR WRITING SESSION ===\n")
        f.write(f"Time: {time.ctime()}\n\n")
        for i, t in enumerate(recognized_history, 1):
            f.write(f"{i}. {t}\n")
    print(f"[SAVE] Saved to {filename}")

def save_undo_step():
    """Save canvas snapshot for undo."""
    global undo_stack
    if canvas is not None:
        undo_stack.append(canvas.copy())
        if len(undo_stack) > 20:
            undo_stack.pop(0)

def undo_action():
    """Revert to last canvas state."""
    global canvas
    if undo_stack:
        canvas = undo_stack.pop()
        print("[UNDO] Reverted.")
    else:
        print("[UNDO] Nothing to undo.")

# ============================================================
# UI DRAWING
# ============================================================

def draw_toolbar(img):
    """Top toolbar with color swatches and buttons."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W_CAM, TOOLBAR_HEIGHT), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.putText(img, "AIR WRITING 2.0", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    start_x = 180
    for i, p in enumerate(PALETTE):
        cx = start_x + i * (SWATCH_SIZE + SWATCH_MARGIN)
        cy = 15

        border = (255, 255, 255) if i == active_color_idx else (80, 80, 80)
        thick = 2 if i == active_color_idx else 1
        cv2.rectangle(img, (cx - 2, cy - 2), (cx + SWATCH_SIZE + 2, cy + SWATCH_SIZE + 2),
                      border, thick, cv2.LINE_AA)

        if p["name"] == "ERASER":
            cv2.rectangle(img, (cx, cy), (cx + SWATCH_SIZE, cy + SWATCH_SIZE), (200, 200, 200), 1)
            cv2.putText(img, "E", (cx + 12, cy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (cx, cy), (cx + SWATCH_SIZE, cy + SWATCH_SIZE), p["color"], cv2.FILLED)

        label_col = (255, 255, 255) if i == active_color_idx else (150, 150, 150)
        cv2.putText(img, p["name"], (cx, cy + SWATCH_SIZE + 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, label_col, 1, cv2.LINE_AA)

    # Buttons
    clear_x = start_x + len(PALETTE) * (SWATCH_SIZE + SWATCH_MARGIN) + 30
    cv2.rectangle(img, (clear_x, 15), (clear_x + 80, 55), (40, 40, 180), cv2.FILLED)
    cv2.putText(img, "CLEAR", (clear_x + 12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ocr_x = clear_x + 110
    cv2.rectangle(img, (ocr_x, 15), (ocr_x + 110, 55), (40, 140, 40), cv2.FILLED)
    cv2.putText(img, "RECOGNIZE", (ocr_x + 8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    undo_x = ocr_x + 130
    cv2.rectangle(img, (undo_x, 15), (undo_x + 80, 55), (100, 100, 100), cv2.FILLED)
    cv2.putText(img, "UNDO", (undo_x + 15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def draw_footer(img, text, status):
    """Bottom bar with status and recognized text."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, H_CAM - FOOTER_HEIGHT), (W_CAM, H_CAM), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    colors = {"WRITING": (0, 255, 100), "RECOGNIZING": (100, 100, 255), "ERASING": (0, 150, 255)}
    sc = colors.get(status, (180, 180, 180))
    cv2.circle(img, (25, H_CAM - FOOTER_HEIGHT // 2), 8, sc, cv2.FILLED)
    cv2.putText(img, status, (40, H_CAM - FOOTER_HEIGHT // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1, cv2.LINE_AA)

    if text:
        cv2.putText(img, f">> {text}", (160, H_CAM - FOOTER_HEIGHT // 2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 50), 4, cv2.LINE_AA)
        cv2.putText(img, f">> {text}", (160, H_CAM - FOOTER_HEIGHT // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2, cv2.LINE_AA)

    cv2.putText(img, "Point=Draw | Two fingers=Hover | Open palm=Clear | 's'=Save | 'q'=Quit",
                (15, H_CAM - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, (100, 100, 100), 1, cv2.LINE_AA)
    return img

def draw_history(img):
    """Sidebar showing last 10 recognized texts."""
    sw = 250
    overlay = img.copy()
    cv2.rectangle(overlay, (W_CAM - sw, TOOLBAR_HEIGHT), (W_CAM, H_CAM - FOOTER_HEIGHT), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, "HISTORY", (W_CAM - sw + 10, TOOLBAR_HEIGHT + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)

    y = TOOLBAR_HEIGHT + 60
    recent = recognized_history[-10:]
    for txt in reversed(recent):
        display = (txt[:22] + "..") if len(txt) > 22 else txt
        cv2.putText(img, f"- {display}", (W_CAM - sw + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 25
    return img

def check_toolbar_selection(x, y):
    """Check if fingertip is over a toolbar button."""
    global active_color_idx, draw_color, brush_thickness, canvas
    global is_actively_writing, recognized_text, interaction_last_time

    if y > TOOLBAR_HEIGHT:
        return False

    now = time.time()
    if now - interaction_last_time < 0.5:
        return False
    interaction_last_time = now

    start_x = 180
    for i, p in enumerate(PALETTE):
        cx = start_x + i * (SWATCH_SIZE + SWATCH_MARGIN)
        if cx <= x <= cx + SWATCH_SIZE and 15 <= y <= 15 + SWATCH_SIZE:
            active_color_idx = i
            draw_color = p["color"]
            brush_thickness = p["thickness"]
            print(f"[UI] Selected: {p['name']}")
            return True

    clear_x = start_x + len(PALETTE) * (SWATCH_SIZE + SWATCH_MARGIN) + 30
    if clear_x <= x <= clear_x + 80 and 15 <= y <= 55:
        save_undo_step()
        canvas = np.zeros_like(canvas)
        is_actively_writing = False
        recognized_text = ""
        print("[UI] Canvas cleared")
        return True

    ocr_x = clear_x + 110
    if ocr_x <= x <= ocr_x + 110 and 15 <= y <= 55:
        perform_recognition(canvas)
        is_actively_writing = False
        return True

    undo_x = ocr_x + 130
    if undo_x <= x <= undo_x + 80 and 15 <= y <= 55:
        undo_action()
        return True

    return False

# ============================================================
# MAIN LOOP
# ============================================================
print("\n===== Air Handwriting Recognition System =====")
print("CONTROLS:")
print("  Index Finger UP        -> Draw")
print("  Index + Middle UP      -> Move / Hover")
print("  Open Palm (All UP)     -> Clear Canvas")
print("  Wait 2.5s idle         -> Auto Recognition")
print("  Press 'q'              -> Quit")
print("  Press 's'              -> Save Session")
print("  Press 'z'              -> Undo")
print("===============================================\n")

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # Hand tracking
    img = tracker.find_hands(img, draw=False)
    lm_list = tracker.find_position(img, draw=False)

    currently_writing = False
    current_status = "IDLE"

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]  # Index fingertip
        fingers = tracker.fingers_up()
        total_up = sum(fingers)

        # --- CLEAR MODE (All 5 fingers UP) ---
        if total_up == 5:
            if is_actively_writing or np.any(canvas != 0):
                save_undo_step()
            canvas = np.zeros_like(img)
            xp, yp = 0, 0
            is_actively_writing = False
            is_drawing_stroke = False
            recognized_text = ""
            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()
            current_status = "CLEARING"

        # --- CONTINUE AN ACTIVE STROKE (UNCONDITIONAL) ---
        # Once drawing starts, we track the fingertip position
        # regardless of what fingers_up() says.
        # Stop ONLY when: hand disappears OR all-5 clear gesture.
        elif is_drawing_stroke:
            currently_writing = True
            is_actively_writing = True
            last_writing_time = time.time()
            current_status = "WRITING" if draw_color != (0, 0, 0) else "ERASING"

            sx, sy = smooth_point(x1, y1)
            pt_color = (255, 255, 255) if draw_color == (0, 0, 0) else draw_color
            cv2.circle(img, (sx, sy), brush_thickness // 2 + 2, pt_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = sx, sy

            cv2.line(canvas, (xp, yp), (sx, sy), draw_color, brush_thickness, cv2.LINE_AA)
            xp, yp = sx, sy

        # --- START A NEW STROKE (Index up, middle+ring down) ---
        elif fingers[1] and not fingers[2] and not fingers[3]:
            save_undo_step()
            is_drawing_stroke = True
            currently_writing = True
            is_actively_writing = True
            last_writing_time = time.time()
            current_status = "WRITING"

            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()
            sx, sy = smooth_point(x1, y1)
            pt_color = (255, 255, 255) if draw_color == (0, 0, 0) else draw_color
            cv2.circle(img, (sx, sy), brush_thickness // 2 + 2, pt_color, cv2.FILLED)
            xp, yp = sx, sy

        # --- SELECTION MODE (Index + Middle, no active stroke) ---
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            xp, yp = 0, 0
            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()
            current_status = "SELECTING"
            cv2.circle(img, (x1, y1), 15, draw_color, 2, cv2.LINE_AA)
            check_toolbar_selection(x1, y1)

        else:
            xp, yp = 0, 0
            is_drawing_stroke = False
            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()

    # Hand disappeared — end stroke
    else:
        if is_drawing_stroke:
            is_drawing_stroke = False
            xp, yp = 0, 0
            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()

    # --- Auto-Recognition on Timeout ---
    if is_actively_writing and not currently_writing:
        elapsed = time.time() - last_writing_time
        if elapsed > TIMEOUT_DURATION:
            current_status = "RECOGNIZING"
            perform_recognition(canvas)
            is_actively_writing = False

    # --- Merge Canvas onto Camera ---
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # --- Draw UI ---
    img = draw_toolbar(img)
    img = draw_history(img)
    img = draw_footer(img, recognized_text, current_status)

    # --- Show ---
    cv2.imshow("Air Handwriting Recognition", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_session()
    elif key == ord('z'):
        undo_action()

cap.release()
cv2.destroyAllWindows()

if recognized_history:
    print("\n=== SESSION HISTORY ===")
    for i, txt in enumerate(recognized_history, 1):
        print(f"  {i}. {txt}")
