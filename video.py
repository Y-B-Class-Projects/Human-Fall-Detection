from fall_core import (
    get_pose_model, get_pose, prepare_image, fall_detection,
    falling_alarm, prepare_vid_out
)
import cv2
import os
from tqdm import tqdm


def process_video_file(video_path, output_dir):
    vid_cap = cv2.VideoCapture(video_path)
    if not vid_cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    model, device = get_pose_model()
    vid_out = prepare_vid_out(video_path, vid_cap, output_dir)

    success, frame = vid_cap.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = vid_cap.read()

    for image in tqdm(frames, desc=f"Processing {os.path.basename(video_path)}"):
        image, output = get_pose(image, model, device)
        _image = prepare_image(image)
        is_fall, bbox = fall_detection(output)

        if is_fall:
            falling_alarm(_image, bbox)
        vid_out.write(_image[:,:,::-1])

    vid_out.release()
    vid_cap.release()


if __name__ == "__main__":
    videos_path = "fall_dataset/videos"
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        process_video_file(video_path, output_dir)
