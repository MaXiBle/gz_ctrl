# core/gaze_tracker.py

import cv2
import mediapipe as mp
import numpy as np

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def get_gaze_point(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None  # ← ВАЖНО: возвращаем None, если нет лица

        lm = results.multi_face_landmarks[0].landmark

        # Зрачки
        left_eye = lm[468]
        right_eye = lm[473]

        # Центр лица: нос + подбородок
        nose_tip = lm[1]
        chin = lm[175]
        face_center_x = (nose_tip.x + chin.x) / 2
        face_center_y = (nose_tip.y + chin.y) / 2

        # Относительное положение глаз от центра лица
        left_gx = left_eye.x - face_center_x
        left_gy = left_eye.y - face_center_y
        right_gx = right_eye.x - face_center_x
        right_gy = right_eye.y - face_center_y

        rel_gx = (left_gx + right_gx) / 2
        rel_gy = (left_gy + right_gy) / 2

        # Нормализация в [0, 1] (эмпирический диапазон)
        max_offset = 0.06  # подобрано экспериментально

        normalized_gx = 0.5 + rel_gx / (2 * max_offset)
        normalized_gy = 0.5 + rel_gy / (2 * max_offset)

        normalized_gx = np.clip(normalized_gx, 0.0, 1.0)
        normalized_gy = np.clip(normalized_gy, 0.0, 1.0)

        return float(normalized_gx), float(normalized_gy)

    def close(self):
        self.face_mesh.close()