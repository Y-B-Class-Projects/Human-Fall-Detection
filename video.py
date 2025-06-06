from fall_core import FallDetectorMulti
from config import FPS, WINDOW_SIZE, V_THRESH, DY_THRESH, ASPECT_RATIO_THRESH
import os


def process_video():
    if os.environ.get("CI_MODE") == "1":
        videos_path = "fall_dataset/ci_videos"
        print("[CI MODE] Only running on CI test videos...")
    else:
        videos_path = "fall_dataset/videos"

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    detector = FallDetectorMulti(
        fps=FPS,
        window_size=WINDOW_SIZE,
        v_thresh=V_THRESH,
        dy_thresh=DY_THRESH,
        ar_thresh=ASPECT_RATIO_THRESH,
    )

    for video in os.listdir(videos_path):
        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        video_path = os.path.join(videos_path, video)
        detector.process_video_file(video_path, output_dir)


if __name__ == "__main__":
    process_video()
