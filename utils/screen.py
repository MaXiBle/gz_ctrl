# utils/screen.py

import pyautogui

def get_screen_size():
    return pyautogui.size()

def generate_calibration_points(screen_w, screen_h, cols=3, rows=3):
    points = []
    for row in range(rows):
        for col in range(cols):
            x = int(col * screen_w / (cols - 1)) if cols > 1 else screen_w // 2
            if row == 0:
                y = int(screen_h * 0.08)   # Верх: 8% (не край)
            elif row == rows - 1:
                y = int(screen_h * 0.92)   # Низ: 92%
            else:
                y = int(screen_h * 0.5)    # Центр
            points.append((x, y))
    return points