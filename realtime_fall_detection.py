import cv2
import torch
import numpy as np
import math
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


# Load the model
def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = attempt_load("yolov7-w6-pose.pt", map_location=device)
    model.eval()

    if torch.cuda.is_available():
        model = model.half().to(device)
    else:
        model = model.float().to(device)

    return model, device


# Preprocess the image to be compatible with YOLO
def preprocess_image(image, model, device):
    img_size = 640
    image = letterbox(image, img_size, stride=32, auto=False)[0]  # Fixed: auto=False
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).to(device)
    image = (
        image.half()
        if next(model.parameters()).dtype == torch.float16
        else image.float()
    )
    image /= 255.0

    if image.ndimension() == 3:
        image = image.unsqueeze(0)
    return image


# Logic to detect a fall based on keypoints
def detect_fall(poses):
    for pose in poses:
        keypoints = torch.tensor(pose[7:]).view(-1, 3)  # Fixed line

        try:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
        except IndexError:
            continue

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2
        ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        shoulder_hip_dist = shoulder_y - hip_y
        hip_ankle_dist = hip_y - ankle_y

        # Simple threshold values (tune them as needed)
        if shoulder_hip_dist < 40 and hip_ankle_dist < 40:
            return True, pose[:4]

    return False, None


# Draw a bounding box and alert text
def draw_fall_alert(image, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    cv2.putText(
        image,
        "Fall Detected!",
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
    )


# Main function
def main():
    model, device = get_pose_model()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = preprocess_image(frame, model, device)

        with torch.no_grad():
            output, _ = model(image)

        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=model.yaml["nc"],
            nkpt=model.yaml["nkpt"],
            kpt_label=True,
        )
        output = output_to_keypoint(output)

        is_fall, bbox = detect_fall(output)
        if is_fall:
            draw_fall_alert(frame, bbox)

        cv2.imshow("Real-Time Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
