import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import telepot
import os
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def fall_detection(idx, nimg):
    xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
    xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)
    left_shoulder_y = output[idx][23]
    left_shoulder_x = output[idx][22]
    right_shoulder_y = output[idx][26]
    left_body_y = output[idx][41]
    left_body_x = output[idx][40]
    right_body_y = output[idx][44]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
    left_foot_y = output[idx][53]
    right_foot_y = output[idx][56]
    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
        cv2.rectangle(nimg, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255),
                      thickness=5, lineType=cv2.LINE_AA)
        cv2.putText(nimg, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)
        print("Person Fall Detected")
    return nimg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)
weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model = model.half().to(device)

video_path = 'Mydata/video_1.mp4'
# pass video to videocapture object
cap = cv2.VideoCapture(video_path)

# check if videocapture not opened
if not cap.isOpened():
    print('Error while trying to read video. Please check path again')

# code to write a video
vid_write_image = letterbox(cap.read()[1], 960, stride=64, auto=True)[0]
resize_height, resize_width = vid_write_image.shape[:2]
out_video_name = f"{video_path.split('/')[-1].split('.')[0]}_keypoint.mp4"
out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))

# count no of frames
frame_count = 0
# count total fps
total_fps = 0

# loop until cap opened or video not complete
while cap.isOpened:
    print("Frame {} Processing".format(frame_count))
    frame_count += 1
    # get frame and success from video capture
    success, image = cap.read()
    # if success is true, means frame exist
    if success:
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(device)
        with torch.no_grad():
            output, _ = model(image)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                         kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
            nimg = fall_detection(idx, nimg)

        # plt.figure(figsize=(8, 8))
        # plt.axis('off')
        # plt.imshow(nimg)
        # plt.savefig('Mydata/frame_{}.jpg'.format(frame_count))

        out.write(nimg)
    else:
        break

# cv2.destroyAllWindows()
out.release()
cap.release()
