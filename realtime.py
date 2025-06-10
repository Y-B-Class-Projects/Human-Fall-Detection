import cv2
import time
from fall_core import FallDetectorMulti
from config import FPS, WINDOW_SIZE, V_THRESH, DY_THRESH, ASPECT_RATIO_THRESH


def process_realtime_camera():
    detector = FallDetectorMulti(
        window_size=WINDOW_SIZE,
        fps=FPS,
        v_thresh=V_THRESH,
        dy_thresh=DY_THRESH,
        ar_thresh=ASPECT_RATIO_THRESH,
    )
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _image, prev_time = detector.handle_frame(frame, prev_time)

        cv2.imshow("Real-Time Fall Detection", _image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_realtime_camera()
