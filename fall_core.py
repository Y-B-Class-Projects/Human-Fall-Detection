import torch
import math
import cv2
import numpy as np
import os
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    weights = torch.load("yolov7-w6-pose.pt", map_location=device, weights_only=False)
    model = weights["model"]
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    return model, device


def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(
        output, 0.25, 0.65, nc=model.yaml["nc"], nkpt=model.yaml["nkpt"], kpt_label=True
    )
    with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output


def prepare_image(image):
    _image = image[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    return _image


def prepare_vid_out(video_path, vid_cap, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    success, first_frame = vid_cap.read()
    if not success:
        raise RuntimeError(f"Failed to read first frame for output setup: {video_path}")

    # Reset video capture position to beginning
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vid_write_image = letterbox(first_frame, 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    video_name = os.path.basename(video_path).split('.')[0] + "_output.mp4"
    out_video_name = os.path.join(output_dir, video_name)
    print(f"[INFO] Writing output to: {out_video_name}")

    out = cv2.VideoWriter(
        out_video_name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (resize_width, resize_height),
    )
    return out


def fall_detection(poses):
    for pose in poses:
        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]
        len_factor = math.sqrt(
            (left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2
        )
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if (
            left_shoulder_y > left_foot_y - len_factor
            and left_body_y > left_foot_y - (len_factor / 2)
            and left_shoulder_y > left_body_y - (len_factor / 2)
            or (
                right_shoulder_y > right_foot_y - len_factor
                and right_body_y > right_foot_y - (len_factor / 2)
                and right_shoulder_y > right_body_y - (len_factor / 2)
            )
            or difference < 0
        ):
            return True, (xmin, ymin, xmax, ymax)
    return False, None


def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(
        image,
        (int(x_min), int(y_min)),
        (int(x_max), int(y_max)),
        color=(255, 0, 0),
        thickness=5,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Person Fell down",
        (11, 100),
        0,
        1,
        [2550, 0, 0],
        thickness=3,
        lineType=cv2.LINE_AA,
    )


def draw_fps(frame, prev_time):
    import time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame, curr_time

