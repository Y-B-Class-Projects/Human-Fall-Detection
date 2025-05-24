from fall_core import (
    get_pose_model, get_pose, prepare_image,
    fall_detection, falling_alarm, draw_fps,
)
import cv2
import time
import torch


def process_realtime_camera():
    model, device = get_pose_model()
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, prev_time = draw_fps(frame, prev_time)
        image_tensor, output = get_pose(frame, model, device)
        _image = prepare_image(image_tensor)
        is_fall, bbox = fall_detection(output)

        if is_fall:
            falling_alarm(_image, bbox)

        cv2.imshow("Real-Time Fall Detection", _image[:,:,::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_realtime_camera()
