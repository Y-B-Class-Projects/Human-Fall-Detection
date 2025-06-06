from fall_core import (
    get_pose_model,
    get_pose,
    prepare_image,
    fall_detection,
    falling_alarm,
    prepare_vid_out,
)
import cv2
import os
from tqdm import tqdm
from collections import deque
from config import WINDOW_SIZE, FPS, V_THRESH, ASPECT_RATIO_THRESH, DY_THRESH


def process_video_file(video_path, output_dir):
    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    model, device = get_pose_model()
    vid_out = prepare_vid_out(video_path, vid_cap, output_dir)
    pose_window = deque(maxlen=WINDOW_SIZE)

    success, frame = vid_cap.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = vid_cap.read()

    for image in tqdm(frames, desc=f"Processing {os.path.basename(video_path)}"):
        image, output = get_pose(image, model, device)
        _image = prepare_image(image)
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
                # debug
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
        vid_out.write(_image[:, :, ::-1])

    vid_out.release()
    vid_cap.release()


if __name__ == "__main__":
    if os.environ.get("CI_MODE") == "1":
        videos_path = "fall_dataset/ci_videos"
        print("[CI MODE] Only running on CI test videos...")
    else:
        videos_path = "fall_dataset/videos"
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        process_video_file(video_path, output_dir)
