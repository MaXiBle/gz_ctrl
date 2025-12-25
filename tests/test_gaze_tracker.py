import unittest
from unittest.mock import Mock, patch
import numpy as np
from core.gaze_tracker import GazeTracker


class TestGazeTracker(unittest.TestCase):
    def setUp(self):
        # Создаем мок для MediaPipe, чтобы избежать загрузки в тестах
        # В новой версии MediaPipe структура изменилась, используем подходящий путь
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_face_mesh_instance = Mock()
            mock_face_mesh_class.return_value = mock_face_mesh_instance
            
            # Мокаем cv2.cvtColor, чтобы избежать проблем с OpenCV
            with patch('cv2.cvtColor') as mock_cvt_color:
                mock_cvt_color.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Теперь создаем экземпляр GazeTracker
                self.mock_face_mesh_instance = mock_face_mesh_instance
                self.tracker = GazeTracker()

    def test_get_gaze_point_no_face_detected(self):
        """Тест: возвращается None, если лицо не обнаружено"""
        # Мокаем результаты обработки кадра
        mock_results = Mock()
        mock_results.multi_face_landmarks = None
        self.mock_face_mesh_instance.process.return_value = mock_results

        # Создаем фиктивный кадр
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Вызываем метод
        with patch('cv2.cvtColor') as mock_cvt_color:
            mock_cvt_color.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            result = self.tracker.get_gaze_point(frame)

        # Проверяем, что возвращается None
        self.assertIsNone(result)

    def test_init_with_custom_parameters(self):
        """Тест: инициализация с параметрами из настроек"""
        # Проверяем, что gaze_offset_max установлен правильно
        # (предполагая, что в настройках GAZE_OFFSET_MAX = 0.06)
        # Нужно создать трекер заново с моками, чтобы параметры были установлены
        with patch('mediapipe.solutions.face_mesh.FaceMesh') as mock_face_mesh_class:
            mock_face_mesh_instance = Mock()
            mock_face_mesh_class.return_value = mock_face_mesh_instance
            
            with patch('cv2.cvtColor') as mock_cvt_color:
                mock_cvt_color.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
                
                tracker = GazeTracker()
                self.assertEqual(tracker.gaze_offset_max, 0.06)

    def tearDown(self):
        self.tracker.close()


if __name__ == '__main__':
    unittest.main()