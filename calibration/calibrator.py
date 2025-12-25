# calibration/calibrator.py

import cv2
import time
import numpy as np
from utils.camera import Camera
from core.gaze_tracker import GazeTracker
from utils.screen import get_screen_size, generate_calibration_points
from config.settings import CALIBRATION_GRID, CAMERA_DEVICE_ID

class Calibrator:
    def __init__(self):
        self.screen_w, self.screen_h = get_screen_size()
        self.calibration_points = generate_calibration_points(
            self.screen_w, self.screen_h,
            cols=CALIBRATION_GRID[0],
            rows=CALIBRATION_GRID[1]
        )
        self.camera = Camera(device_id=CAMERA_DEVICE_ID)
        self.gaze_tracker = GazeTracker()
        self.gaze_samples = []
        self.screen_points = []

    def start(self):
        print(f"Начинается калибровка на экране {self.screen_w}x{self.screen_h}...")
        for i, (sx, sy) in enumerate(self.calibration_points):
            print(f"→ Точка {i+1}/{len(self.calibration_points)}: ({sx}, {sy})")

            frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.circle(frame, (sx, sy), 25, (0, 0, 255), -1)
            cv2.namedWindow("Калибровка", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Калибровка", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Калибровка", frame)
            cv2.waitKey(500)

            samples = []
            start_time = time.time()
            while time.time() - start_time < 2.0:
                img = self.camera.get_frame()
                if img is None:
                    continue

                gaze = self.gaze_tracker.get_gaze_point(img)
                if gaze:
                    gx, gy = gaze
                    # === ВРЕМЕННО: ослабленный фильтр для отладки ===
                    if 0.0 <= gx <= 1.0 and 0.0 <= gy <= 1.0:
                        samples.append([gx, gy])

                    # Отладка: показываем глаза
                    h, w = img.shape[:2]
                    cv2.circle(img, (int(gx * w), int(gy * h)), 5, (0, 255, 0), -1)
                    cv2.imshow("Отладка калибровки", img)
                cv2.waitKey(1)

            if samples:
                avg_gaze = np.mean(samples, axis=0)
                self.gaze_samples.append(avg_gaze.tolist())
                self.screen_points.append([sx, sy])
                print(f"   → Записано: ({avg_gaze[0]:.3f}, {avg_gaze[1]:.3f})")
            else:
                print("   → ❌ Нет данных! Убедитесь, что лицо видно.")

            cv2.waitKey(500)

        cv2.destroyWindow("Калибровка")
        cv2.destroyWindow("Отладка калибровки")
        self.save_calibration()

    def save_calibration(self):
        if len(self.gaze_samples) == 0:
            print("❌ Калибровка не содержит данных! Запустите заново.")
            return

        from core.screen_mapper import ScreenMapper
        mapper = ScreenMapper(screen_w=self.screen_w, screen_h=self.screen_h)
        mapper.save_calibration(np.array(self.gaze_samples), np.array(self.screen_points))
        print("✅ Калибровка сохранена.")