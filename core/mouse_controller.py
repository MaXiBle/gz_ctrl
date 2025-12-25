# core/mouse_controller.py
import time

import pyautogui
import numpy as np
from config.settings import DWELL_TIME, SMOOTHING_WINDOW

class MouseController:
    def __init__(self, dwell_time=DWELL_TIME, smoothing_window=SMOOTHING_WINDOW):
        self.dwell_time = dwell_time
        self.last_gaze_x = None
        self.last_gaze_y = None
        self.fixation_start_time = None
        self.smoothing_window = smoothing_window
        self.screen_x_history = []
        self.screen_y_history = []
        pyautogui.FAILSAFE = False

    def _smooth_position(self, screen_x, screen_y):
        self.screen_x_history.append(screen_x)
        self.screen_y_history.append(screen_y)
        if len(self.screen_x_history) > self.smoothing_window:
            self.screen_x_history.pop(0)
            self.screen_y_history.pop(0)
        return int(np.mean(self.screen_x_history)), int(np.mean(self.screen_y_history))

    def update_cursor(self, screen_x, screen_y):
        smoothed_x, smoothed_y = self._smooth_position(screen_x, screen_y)
        pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)

    def handle_dwell_click(self, gaze_x, gaze_y):
        if self.last_gaze_x is None:
            self.last_gaze_x, self.last_gaze_y = gaze_x, gaze_y
            self.fixation_start_time = time.time()
        else:
            move_dist = abs(gaze_x - self.last_gaze_x) + abs(gaze_y - self.last_gaze_y)
            if move_dist < 0.01:
                if self.fixation_start_time and (time.time() - self.fixation_start_time) >= self.dwell_time:
                    pyautogui.click()
                    self.fixation_start_time = time.time() + 10
            else:
                self.fixation_start_time = time.time()
                self.last_gaze_x, self.last_gaze_y = gaze_x, gaze_y