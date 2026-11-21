import cv2
import threading

class CameraStream:
    def __init__(self, src, name):
        self.name = name
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) \
                   if isinstance(src, int) else \
                   cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()


# --- STEP 5a: Initialize all 3 cameras ---
cam0 = CameraStream(0, "Built-in")
cam1 = CameraStream(1, "USB")
cam2 = CameraStream("http://192.168.1.2:8080/video", "Phone")

cameras = [cam0, cam1, cam2]

# --- STEP 5b: Stream and display ---
while True:
    for cam in cameras:
        frame = cam.read()
        if frame is not None:
            cv2.imshow(cam.name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- STEP 5c: Release all ---
for cam in cameras:
    cam.stop()
cv2.destroyAllWindows()