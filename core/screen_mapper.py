# core/screen_mapper.py

import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.screen import get_screen_size

class ScreenMapper:
    def __init__(self, calibration_file="calibration/calibration_data.json", screen_w=None, screen_h=None):
        self.calibration_file = calibration_file
        self.screen_w = screen_w or get_screen_size()[0]
        self.screen_h = screen_h or get_screen_size()[1]
        self.model_x = None
        self.model_y = None
        self.load_calibration()

    def load_calibration(self):
        if not os.path.exists(self.calibration_file):
            return

        try:
            with open(self.calibration_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    print("⚠️ Файл калибровки пуст.")
                    return
                data = json.loads(content)
                gaze_coords = np.array(data.get("gaze_coords", []))
                screen_coords = np.array(data.get("screen_coords", []))

                if len(gaze_coords) == 0 or len(screen_coords) == 0:
                    print("⚠️ Калибровка не содержит данных.")
                    return

                # Убедимся, что массивы 2D
                if gaze_coords.ndim == 1:
                    gaze_coords = gaze_coords.reshape(-1, 2)
                if screen_coords.ndim == 1:
                    screen_coords = screen_coords.reshape(-1, 2)

                self.model_x = LinearRegression().fit(gaze_coords[:, 0].reshape(-1, 1), screen_coords[:, 0])
                self.model_y = LinearRegression().fit(gaze_coords[:, 1].reshape(-1, 1), screen_coords[:, 1])
        except Exception as e:
            print(f"⚠️ Ошибка загрузки калибровки: {e}")

    def save_calibration(self, gaze_coords, screen_coords):
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        data = {
            "gaze_coords": gaze_coords.tolist(),
            "screen_coords": screen_coords.tolist()
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)

    def map_to_screen(self, gaze_x, gaze_y):
        if not (0.0 < gaze_x < 1.0 and 0.0 < gaze_y < 1.0):
            return self.screen_w // 2, self.screen_h // 2

        if self.model_x is not None and self.model_y is not None:
            screen_x = int(self.model_x.predict([[gaze_x]])[0])
            screen_y = int(self.model_y.predict([[gaze_y]])[0])
        else:
            screen_x = int(gaze_x * self.screen_w)
            screen_y = int(gaze_y * self.screen_h)

        screen_x = max(0, min(screen_x, self.screen_w - 1))
        screen_y = max(0, min(screen_y, self.screen_h - 1))
        return screen_x, screen_y