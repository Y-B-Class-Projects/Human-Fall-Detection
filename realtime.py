from fall_core import (
    get_pose_model,
    get_pose,
    prepare_image,
    fall_detection,
    falling_alarm,
    draw_fps,
)
import cv2
import time
import torch
from collections import deque
from config import WINDOW_SIZE, FPS, V_THRESH, ASPECT_RATIO_THRESH, DY_THRESH


def process_realtime_camera():
    model, device = get_pose_model()
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    pose_window = deque(maxlen=WINDOW_SIZE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, prev_time = draw_fps(frame, prev_time)
        image_tensor, output = get_pose(frame, model, device)
        _image = prepare_image(image_tensor)
        if len(output) > 0:
            pose_window.append(output)
            if len(pose_window) == WINDOW_SIZE:
                is_fall, bbox, debug_text, tag = fall_detection(
                    pose_window,
                    WINDOW_SIZE,
                    FPS,
                    V_THRESH,
                    ASPECT_RATIO_THRESH,
                    DY_THRESH,
                )
                if debug_text:
                    _image = cv2.putText(
                        _image,
                        f"{tag}: {debug_text}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                if is_fall:
                    falling_alarm(_image, bbox)

        cv2.imshow("Real-Time Fall Detection", _image[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_realtime_camera()
