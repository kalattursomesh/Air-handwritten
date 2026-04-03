import cv2

def test_cameras():
    print("Checking for available cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Camera {i} found! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"[WARN] Camera {i} found but failed to read frames.")
            cap.release()
        else:
            print(f"[-] Camera {i} not found.")

if __name__ == "__main__":
    test_cameras()
