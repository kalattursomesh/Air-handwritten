import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """
    Hand tracking using MediaPipe Tasks API (HandLandmarker).
    Clean, reliable implementation — no frame skipping or aggressive resizing.
    """

    MODEL_PATH = "hand_landmarker.task"
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, max_hands=1, detection_con=0.7, track_con=0.5):
        self.max_hands = max_hands

        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=track_con,
            min_tracking_confidence=track_con,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.lm_list = []
        self.handedness = "Right"
        self._results = None
        self._frame_timestamp = 0

    def find_hands(self, img, draw=True):
        """Process a BGR frame and detect hands."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self._frame_timestamp += 33  # ~30 FPS
        self._results = self.landmarker.detect_for_video(mp_image, self._frame_timestamp)

        if draw and self._results and self._results.hand_landmarks:
            self._draw_landmarks(img)
        return img

    def find_position(self, img, hand_no=0, draw=False):
        """Return list of [id, x, y] for each landmark."""
        self.lm_list = []

        if self._results and self._results.hand_landmarks:
            if hand_no < len(self._results.hand_landmarks):
                h, w, _ = img.shape
                landmarks = self._results.hand_landmarks[hand_no]
                for idx, lm in enumerate(landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([idx, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if self._results.handedness:
                    label = self._results.handedness[hand_no][0].category_name
                    self.handedness = "Left" if label == "Right" else "Right"

        return self.lm_list

    def fingers_up(self):
        """Return list of 5 ints (1=up, 0=down) for each finger."""
        fingers = []
        if len(self.lm_list) == 0:
            return fingers

        # Thumb
        if self.handedness == "Right":
            fingers.append(
                1 if self.lm_list[self.TIP_IDS[0]][1] > self.lm_list[self.TIP_IDS[0] - 1][1] else 0
            )
        else:
            fingers.append(
                1 if self.lm_list[self.TIP_IDS[0]][1] < self.lm_list[self.TIP_IDS[0] - 1][1] else 0
            )

        # Four fingers
        for i in range(1, 5):
            fingers.append(
                1 if self.lm_list[self.TIP_IDS[i]][2] < self.lm_list[self.TIP_IDS[i] - 2][2] else 0
            )
        return fingers

    def _draw_landmarks(self, img):
        """Draw hand landmarks and connections."""
        h, w, _ = img.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        for hand_landmarks in self._results.hand_landmarks:
            points = []
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))

            for c in connections:
                if c[0] < len(points) and c[1] < len(points):
                    cv2.line(img, points[c[0]], points[c[1]], (0, 255, 200), 2, cv2.LINE_AA)

            for pt in points:
                cv2.circle(img, pt, 4, (255, 0, 200), cv2.FILLED, cv2.LINE_AA)
