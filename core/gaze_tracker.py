# core/gaze_tracker.py

import cv2
import mediapipe as mp
import numpy as np
from config.settings import FACE_DETECTION_CONFIDENCE, FACE_TRACKING_CONFIDENCE, GAZE_OFFSET_MAX

class GazeTracker:
    def __init__(self):
        self.gaze_offset_max = GAZE_OFFSET_MAX
        
        # Используем FaceLandmarker из новой версии MediaPipe
        # Для автоматической загрузки модели укажем соответствующий путь
        base_options = mp.tasks.BaseOptions(
            model_asset_path=None  # Автоматическая загрузка модели
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def get_gaze_point(self, frame):
        # Конвертируем BGR в RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Создаем MediaPipe Image изображение
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Обрабатываем изображение
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None, None  # ← Возвращаем None для взгляда и None для центра лица

        # Получаем landmark'ы первого лица
        landmarks = results.face_landmarks[0]

        # Зрачки (в новой версии MediaPipe номера точек могут отличаться)
        # В новой версии MediaPipe нет специфических точек для зрачков, используем приближенные
        left_eye = landmarks[468]  # приближенная точка левого глаза
        right_eye = landmarks[473]  # приближенная точка правого глаза

        # Центр лица: нос + подбородок
        nose_tip = landmarks[1]
        chin = landmarks[175]
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
        max_offset = self.gaze_offset_max

        normalized_gx = 0.5 + rel_gx / (2 * max_offset)
        normalized_gy = 0.5 + rel_gy / (2 * max_offset)

        normalized_gx = np.clip(normalized_gx, 0.0, 1.0)
        normalized_gy = np.clip(normalized_gy, 0.0, 1.0)

        return (float(normalized_gx), float(normalized_gy)), (face_center_x, face_center_y)

    def close(self):
        # В новой версии MediaPipe закрытие может быть не требуется
        pass