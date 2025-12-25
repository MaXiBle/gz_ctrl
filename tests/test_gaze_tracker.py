import unittest
from unittest.mock import Mock, patch
import numpy as np
from core.gaze_tracker import GazeTracker


class TestGazeTracker(unittest.TestCase):
    def setUp(self):
        # Мокаем все зависимости MediaPipe перед созданием экземпляра
        self.mp_patcher = patch('mediapipe.tasks.vision.FaceLandmarker')
        self.mock_face_landmarker_class = self.mp_patcher.start()
        
        # Создаем mock для FaceLandmarkerOptions
        self.options_patcher = patch('mediapipe.tasks.vision.FaceLandmarkerOptions')
        self.mock_options_class = self.options_patcher.start()
        
        # Создаем mock для BaseOptions
        self.base_options_patcher = patch('mediapipe.tasks.BaseOptions')
        self.mock_base_options_class = self.base_options_patcher.start()
        
        # Мокаем cv2.cvtColor, чтобы избежать проблем с OpenCV
        self.cvt_patcher = patch('cv2.cvtColor')
        self.mock_cvt_color = self.cvt_patcher.start()
        self.mock_cvt_color.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Мокаем mp.Image
        self.image_patcher = patch('mediapipe.Image')
        self.mock_image_class = self.image_patcher.start()
        
        # Создаем экземпляр трекера
        self.tracker = GazeTracker()

    def test_get_gaze_point_no_face_detected(self):
        """Тест: возвращается None, если лицо не обнаружено"""
        # Создаем mock для результатов детекции
        mock_results = Mock()
        mock_results.face_landmarks = []
        
        # Мокаем метод detect
        self.mock_face_landmarker_class.return_value.detect.return_value = mock_results

        # Создаем фиктивный кадр
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Вызываем метод
        result_gaze, result_face_center = self.tracker.get_gaze_point(frame)

        # Проверяем, что возвращается None
        self.assertIsNone(result_gaze)
        self.assertIsNone(result_face_center)

    def test_get_gaze_point_with_face_detected(self):
        """Тест: возвращаются координаты взгляда и центра лица при обнаружении лица"""
        # Создаем mock для landmark'ов
        mock_landmarks = [Mock() for _ in range(478)]
        for lm in mock_landmarks:
            lm.x = 0.5
            lm.y = 0.5
        
        # Устанавливаем специфические значения для ключевых точек
        mock_landmarks[1] = Mock()  # nose_tip
        mock_landmarks[1].x = 0.5
        mock_landmarks[1].y = 0.3
        mock_landmarks[175] = Mock()  # chin
        mock_landmarks[175].x = 0.5
        mock_landmarks[175].y = 0.7
        mock_landmarks[468] = Mock()  # left eye (now used as left eye inner)
        mock_landmarks[468].x = 0.45
        mock_landmarks[468].y = 0.45
        mock_landmarks[473] = Mock()  # right eye (now used as right eye inner)
        mock_landmarks[473].x = 0.55
        mock_landmarks[473].y = 0.45
        mock_landmarks[6] = Mock()  # nose bridge
        mock_landmarks[6].x = 0.5
        mock_landmarks[6].y = 0.4
        
        # Создаем mock для результатов детекции
        mock_results = Mock()
        mock_results.face_landmarks = [mock_landmarks]
        
        # Мокаем метод detect
        self.mock_face_landmarker_class.return_value.detect.return_value = mock_results

        # Создаем фиктивный кадр
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Вызываем метод
        result_gaze, result_face_center = self.tracker.get_gaze_point(frame)

        # Проверяем, что возвращаются корректные значения
        self.assertIsNotNone(result_gaze)
        self.assertIsNotNone(result_face_center)
        self.assertIsInstance(result_gaze, tuple)
        self.assertIsInstance(result_face_center, tuple)
        self.assertEqual(len(result_gaze), 2)
        self.assertEqual(len(result_face_center), 2)
        # Проверяем, что значения в пределах [0, 1]
        self.assertGreaterEqual(result_gaze[0], 0.0)
        self.assertLessEqual(result_gaze[0], 1.0)
        self.assertGreaterEqual(result_gaze[1], 0.0)
        self.assertLessEqual(result_gaze[1], 1.0)

    def test_init_with_custom_parameters(self):
        """Тест: инициализация с параметрами из настроек"""
        # Проверяем, что gaze_offset_max установлен правильно
        self.assertEqual(self.tracker.gaze_offset_max, 0.06)

    def tearDown(self):
        # Останавливаем все патчеры
        self.mp_patcher.stop()
        self.options_patcher.stop()
        self.base_options_patcher.stop()
        self.cvt_patcher.stop()
        self.image_patcher.stop()
        self.tracker.close()


if __name__ == '__main__':
    unittest.main()