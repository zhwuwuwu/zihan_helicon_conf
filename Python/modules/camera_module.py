import cv2

class CameraModule:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera = None
        self.camera_id = camera_id
        self.width, self.height = resolution
        self.is_running = False

    def start(self):
        self.camera = cv2.VideoCapture(self.camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.camera.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.is_running = True

    def stop(self):
        if self.is_running:
            self.camera.release()
            self.is_running = False

    def get_frame(self):
        if not self.is_running:
            return None
        ret, frame = self.camera.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None
    