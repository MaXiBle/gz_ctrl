import cv2

class Camera:
    def __init__(self, device_id=None):
        # Используем значение по умолчанию 0, если device_id не указан
        self.device_id = device_id or 0
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с ID {self.device_id}.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # зеркальное отражение

    def release(self):
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()