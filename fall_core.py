import torch
import math
import cv2
import numpy as np
import os
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def compute_center_of_mass(pose):
    left_shoulder = (pose[10], pose[11])
    right_shoulder = (pose[13], pose[14])
    left_hip = (pose[22], pose[23])
    right_hip = (pose[25], pose[26])
    cx = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
    cy = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
    return cx, cy


def compute_center_velocity(pose_start, pose_end, fps, window_size):
    com_start = compute_center_of_mass(pose_start)
    com_end = compute_center_of_mass(pose_end)
    dx = com_end[0] - com_start[0]
    dy = com_end[1] - com_start[1]
    distance = math.sqrt(dx ** 2 + dy ** 2)
    time_elapsed = (window_size - 1) / fps
    velocity = distance / time_elapsed
    return min(velocity, 300.0), dy


def compute_bbox_aspect_ratio(pose):
    x_vals = [pose[i] for i in range(0, len(pose), 3)]
    y_vals = [pose[i] for i in range(1, len(pose), 3)]
    width = max(x_vals) - min(x_vals)
    height = max(y_vals) - min(y_vals)
    return width / height if height != 0 else 0


def compute_aspect_ratio_delta(pose_start, pose_end):
    ar_start = compute_bbox_aspect_ratio(pose_start)
    ar_end = compute_bbox_aspect_ratio(pose_end)
    return ar_end - ar_start, ar_start, ar_end


def find_most_similar_pose(reference_pose, candidate_poses):
    ref_cx, ref_cy = compute_center_of_mass(reference_pose)
    min_dist = float("inf")
    best_pose = None
    for pose in candidate_poses:
        cx, cy = compute_center_of_mass(pose)
        dist = (ref_cx - cx)**2 + (ref_cy - cy)**2
        if dist < min_dist:
            min_dist = dist
            best_pose = pose
    return best_pose


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


def fall_detection(pose_window, window_size, fps, v_thresh, aspect_ratio_thresh, dy_thresh):
    if len(pose_window) < window_size:
        return False, None, None, None

    pose_start = pose_window[0][0]
    pose_end_all = pose_window[-1]
    pose_end = find_most_similar_pose(pose_start, pose_end_all)

    v, dy = compute_center_velocity(pose_start, pose_end, fps, window_size)
    ar_delta, ar_start, ar_end = compute_aspect_ratio_delta(pose_start, pose_end)

    debug_text = (
        f"v={v:.2f}/{v_thresh:.2f}px/s, "
        f"y={dy:.1f}/{dy_thresh:.1f}, "
        f"ar={ar_delta:.2f}/{aspect_ratio_thresh:.2f}"
    )
    print(f"[TRACE] {debug_text}")
    
    cond_speed_drop = v > v_thresh and dy > dy_thresh
    cond_down_flat = dy > dy_thresh and ar_delta > aspect_ratio_thresh

    if cond_speed_drop or cond_down_flat:
        tag = (
            ("SpeedDrop " if cond_speed_drop else "") +
            ("DownFlat " if cond_down_flat else "")
        ).strip()

        xmin = pose_end[2] - pose_end[4] / 2
        ymin = pose_end[3] - pose_end[5] / 2
        xmax = pose_end[2] + pose_end[4] / 2
        ymax = pose_end[3] + pose_end[5] / 2
        return True, (xmin, ymin, xmax, ymax), debug_text, tag

    return False, None, debug_text, ""


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

