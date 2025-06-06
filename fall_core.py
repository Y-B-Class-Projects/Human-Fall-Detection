import cv2
import os
import numpy as np
from collections import deque
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint
from torchvision import transforms
import torch
import math
import time
from uuid import uuid4


class PersonFallTracker:
    def __init__(self, window_size, fps, v_thresh, ar_thresh, dy_thresh):
        self.pose_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.fps = fps
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh

    def add_pose(self, pose):
        if self.is_pose_complete(pose):
            self.pose_window.append(pose)

    def is_ready(self):
        return len(self.pose_window) == self.window_size

    def compute_center_of_mass(self, pose):
        return np.mean(
            [
                [pose[10], pose[11]],  # left shoulder
                [pose[13], pose[14]],  # right shoulder
                [pose[22], pose[23]],  # left hip
                [pose[25], pose[26]],  # right hip
            ],
            axis=0,
        )

    def compute_velocity(self, p1, p2):
        c1 = self.compute_center_of_mass(p1)
        c2 = self.compute_center_of_mass(p2)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        dist = math.sqrt(dx**2 + dy**2)
        t = (self.window_size - 1) / self.fps
        return min(dist / t, 300.0), dy

    def compute_ar_delta(self, p1, p2):
        def ar(p):
            length = len(p) - (len(p) % 3)
            x = [p[i] for i in range(0, length, 3)]
            y = [p[i + 1] for i in range(0, length, 3)]
            w, h = max(x) - min(x), max(y) - min(y)
            return w / h if h else 0

        return ar(p2) - ar(p1)

    def check_fall(self):
        if not self.is_ready():
            return False, None, None, ""

        p1, p2 = self.pose_window[0], self.pose_window[-1]
        v, dy = self.compute_velocity(p1, p2)
        ar_delta = self.compute_ar_delta(p1, p2)

        ar_start = self._safe_aspect_ratio(p1)
        ar_end = self._safe_aspect_ratio(p2)

        tag = []
        if v > self.v_thresh and dy > self.dy_thresh and ar_end > 0.1:
            tag.append("SpeedDrop")
        if dy > self.dy_thresh and ar_delta > self.ar_thresh:
            tag.append("DownFlat")

        debug = (
            f"v={v:.1f}/{self.v_thresh:.1f}, "
            f"dy={dy:.1f}/{self.dy_thresh:.1f}, "
            f"ar={ar_delta:.2f}/{self.ar_thresh:.2f}"
        )

        if tag:
            cx, cy, w, h = p2[2], p2[3], p2[4], p2[5]
            return (
                True,
                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                debug,
                " ".join(tag),
            )

        return False, None, debug, ""

    def _safe_aspect_ratio(self, p):
        length = len(p) - (len(p) % 3)
        x = [p[i] for i in range(0, length, 3)]
        y = [p[i + 1] for i in range(0, length, 3)]
        w, h = max(x) - min(x), max(y) - min(y)
        return w / h if h else 0

    def is_pose_complete(self, pose, required_joints=(11, 14, 23, 26)):
        try:
            complete = True
            visible_joints = 0
            length = len(pose) - (len(pose) % 3)

            for i in range(0, length, 3):
                x, y, conf = pose[i], pose[i + 1], pose[i + 2]
                if conf > 0.2:
                    visible_joints += 1

            for idx in required_joints:
                if pose[idx] == 0 or pose[idx + 1] == 0:
                    complete = False

            return complete and visible_joints >= 10
        except IndexError:
            return False


class FallDetectorMulti:
    def __init__(
        self,
        model_path="yolov7-w6-pose.pt",
        window_size=10,
        fps=30,
        v_thresh=60.0,
        ar_thresh=0.35,
        dy_thresh=20.0,
    ):
        self.model, self.device = self.load_model(model_path)
        self.trackers = {}
        self.window_size = window_size
        self.fps = fps
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh
        self.next_id = 1

    def draw_debug_overlay(self, image, results):
        for tid, pose, tag, debug, bbox, v, dy, ar in results:
            cx, cy = int(pose[2]), int(pose[3])

            cv2.putText(
                image,
                f"ID: {tid}",
                (cx, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            debug_text = (
                (f"v={v:.1f}/{self.v_thresh:.1f}" if v is not None else "v=N/A")
                + (
                    f", dy={dy:.1f}/{self.dy_thresh:.1f}"
                    if dy is not None
                    else ", dy=N/A"
                )
                + (
                    f", ar={ar:.2f}/{self.ar_thresh:.2f}"
                    if ar is not None
                    else ", ar=N/A"
                )
            )

            cv2.putText(
                image,
                debug_text,
                (cx, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 255, 100),
                1,
            )

        return image

    def load_model(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(path, map_location=device, weights_only=False)
        model = weights["model"].float().eval()
        return (model.half().to(device) if torch.cuda.is_available() else model), device

    def get_pose(self, image):
        image = letterbox(image, 960, stride=64, auto=True)[0]
        tensor = transforms.ToTensor()(image).unsqueeze(0)
        if torch.cuda.is_available():
            tensor = tensor.half().to(self.device)
        with torch.no_grad():
            output, _ = self.model(tensor)
        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=self.model.yaml["nc"],
            nkpt=self.model.yaml["nkpt"],
            kpt_label=True,
        )
        return output_to_keypoint(output), image[:, :, ::-1]

    def match_pose_to_tracker(
        self,
        pose,
        trackers,
        assigned_ids,
        dist_thresh=80,
        height_thresh=60,
        timeout=1.0,
    ):
        import time

        def center_and_height(p):
            length = len(p) - (len(p) % 3)
            keypoints = [(p[i], p[i + 1]) for i in range(0, length - 1, 3)]
            x_vals = [pt[0] for pt in keypoints]
            y_vals = [pt[1] for pt in keypoints]
            center = (np.mean(x_vals), np.mean(y_vals))
            height = max(y_vals) - min(y_vals)
            return center, height

        c_pose, h_pose = center_and_height(pose)
        best_tid = None
        best_dist = float("inf")
        now = time.time()

        for tid, tracker in trackers.items():
            if tid in assigned_ids:
                continue

            if len(tracker.pose_window) == 0:
                continue

            if hasattr(tracker, "last_update") and now - tracker.last_update > timeout:
                continue

            c_track, h_track = center_and_height(tracker.pose_window[-1])
            dist = np.linalg.norm(np.array(c_pose) - np.array(c_track))
            h_diff = abs(h_pose - h_track)

            if dist < dist_thresh and h_diff < height_thresh:
                if dist < best_dist:
                    best_tid = tid
                    best_dist = dist

        if best_tid is not None:
            assigned_ids.add(best_tid)
        return best_tid

    def process_video_file(self, path, out_dir="output_videos"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("[ERR] can't open video:", path)
            return

        os.makedirs(out_dir, exist_ok=True)
        success, first = cap.read()
        vid_shape = letterbox(first, 960, stride=64, auto=True)[0].shape
        out_path = os.path.join(
            out_dir, os.path.basename(path).split(".")[0] + "_output.mp4"
        )
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (vid_shape[1], vid_shape[0])
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            people, processed_frame = self.get_pose(frame)
            _image = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            assigned_ids = set()
            results = []

            for pose in people:
                tid = self.match_pose_to_tracker(pose, self.trackers, assigned_ids)
                if tid is None:
                    tid = str(self.next_id)
                    self.next_id += 1
                    self.trackers[tid] = PersonFallTracker(
                        self.window_size,
                        self.fps,
                        self.v_thresh,
                        self.ar_thresh,
                        self.dy_thresh,
                    )
                self.trackers[tid].add_pose(pose)
                self.trackers[tid].last_update = time.time()

                tag = debug = bbox = None
                v = dy = ar = None

                if self.trackers[tid].is_ready():
                    is_fall, bbox, debug, tag = self.trackers[tid].check_fall()
                    p1, p2 = (
                        self.trackers[tid].pose_window[0],
                        self.trackers[tid].pose_window[-1],
                    )
                    v, dy = self.trackers[tid].compute_velocity(p1, p2)
                    ar = self.trackers[tid].compute_ar_delta(p1, p2)

                    if is_fall and bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                        cv2.putText(
                            _image,
                            "FALL DETECTED",
                            (x1, y1 - 10),
                            0,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

                cx, cy = int(pose[2]), int(pose[3])
                results.append((tid, pose, tag, debug, bbox, v, dy, ar))

            _image = self.draw_debug_overlay(_image, results)
            writer.write(_image)

        cap.release()
        writer.release()
        print(f"[DONE] Saved to {out_path}")

    def draw_fps(self, frame, prev_time):
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

    def handle_frame(self, frame, prev_time=None, writer=None):
        people, processed_frame = self.get_pose(frame)
        _image = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        results = []
        for pose in people:
            tid = self.match_pose_to_tracker(pose, self.trackers)
            if tid is None:
                tid = str(uuid4())[:8]
                self.trackers[tid] = deque(maxlen=self.window_size)
            self.trackers[tid].append(pose)

            tag = debug = bbox = None
            v = dy = ar_delta = None
            if len(self.trackers[tid]) == self.window_size:
                p1, p2 = self.trackers[tid][0], self.trackers[tid][-1]
                v, dy = self.compute_velocity(p1, p2)
                ar_delta = self.compute_ar_delta(p1, p2)

                cond_v = v > self.v_thresh and dy > self.dy_thresh
                cond_ar = dy > self.dy_thresh and ar_delta > self.ar_thresh
                tag_list = []
                if cond_v:
                    tag_list.append("SpeedDrop")
                if cond_ar:
                    tag_list.append("DownFlat")
                tag = " ".join(tag_list)
                debug = f"v={v:.1f}, dy={dy:.1f}, arΔ={ar_delta:.2f}"

                if tag:
                    bbox = (
                        int(p2[2] - p2[4] / 2),
                        int(p2[3] - p2[5] / 2),
                        int(p2[2] + p2[4] / 2),
                        int(p2[3] + p2[5] / 2),
                    )

            results.append((tid, pose, tag, debug, bbox, v, dy, ar_delta))

        _image = self.draw_debug_overlay(_image, results)

        if prev_time is not None:
            _image, new_time = self.draw_fps(_image, prev_time)
            if writer:
                writer.write(_image)
            return _image, new_time
        else:
            if writer:
                writer.write(_image)
            return _image
