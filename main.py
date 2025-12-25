# main.py

import cv2
from config.settings import DWELL_TIME
from core.gaze_tracker import GazeTracker
from core.mouse_controller import MouseController
from core.screen_mapper import ScreenMapper
from utils.camera import Camera
from utils.screen import get_screen_size
from calibration.calibrator import Calibrator

def main():
    print("Запуск Gaze Control...")
    screen_w, screen_h = get_screen_size()
    print(f"Обнаружен экран: {screen_w}x{screen_h}")

    mapper = ScreenMapper(screen_w=screen_w, screen_h=screen_h)
    if mapper.model_x is None or mapper.model_y is None:
        print("Калибровка не найдена или повреждена. Запускаю калибровку...")
        calibrator = Calibrator()
        calibrator.start()
        mapper.load_calibration()

    camera = Camera()
    gaze_tracker = GazeTracker()
    mouse_controller = MouseController(dwell_time=DWELL_TIME)

    print("Управление активно. Нажмите 'q' для выхода.")

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        gaze = gaze_tracker.get_gaze_point(frame)
        if gaze:
            gx, gy = gaze
            screen_x, screen_y = mapper.map_to_screen(gx, gy)
            mouse_controller.update_cursor(screen_x, screen_y)
            mouse_controller.handle_dwell_click(gx, gy)

            # Отладка
            h, w = frame.shape[:2]
            cv2.circle(frame, (int(gx * w), int(gy * h)), 3, (0, 255, 0), -1)

        cv2.imshow("Gaze Control — нажмите 'q'", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    gaze_tracker.close()
    camera.release()
    cv2.destroyAllWindows()
    print("Выход.")

if __name__ == "__main__":
    main()