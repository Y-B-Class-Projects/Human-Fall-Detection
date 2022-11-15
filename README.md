# Human-Fall-Detection

Human Fall Detection Project

In this project, we developed a system using video analysis that can detect when person is falling.

A path to a folder containing video files is provided to the system, which then returns a new video with alarms regarding human falls.

The system scans through each frame of the video and, using the Yolov7 pose model, calculates and generates 17 key - points for each person, each of which represents the location of that person's body parts in that frame.

For example



With TOLOv7

| Accuracy  | Precision | Recall | F1 score |
| ------------- | ------------- | ------------- | ------------- |
|  81.18%  | 83.27% | 83.58%  | 83.42%  |



![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_1_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_2_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_4_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_5_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_6_keypoint.gif)
