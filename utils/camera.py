import cv2

class Camera:
    def __init__(self, device_id=0):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть камеру.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # зеркальное отражение

    def release(self):
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()