import unittest
import cv2
from core.gaze_tracker import GazeTracker

class TestGazeTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = GazeTracker()

    def test_get_gaze_point(self):
        # Это пример — нужно тестировать с реальным кадром
        # Для юнит-тестов лучше использовать mock-объекты
        pass

    def tearDown(self):
        self.tracker.close()

if __name__ == '__main__':
    unittest.main()