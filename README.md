# Human-Fall-Detection

## The Project

In this project, we developed a video analysis system that can detect events of human falling.

The system receives video as input, scans each frame of the video, and then creates 17 key-points for each individual, each of which corresponds to a position of that person's body parts in that frame. This is done using the [YOLOv7-POSE](https://github.com/WongKinYiu/yolov7/tree/pose "YOLOv7-POSE") model.

For example

![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/Mydata/keypoints-example.png)

You can learn more about YOLOv7-POSE by reading this [document](https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf "document").

We initially created an LSTM-based neural network that learned from a database of photos of humans that were categorized as "falling" and "not falling."
We obtained roughly 500 images from the Internet for this purpose.
After we used the data to train the model, we discovered that the network did not succeed in learning, and we got poor results using the test images.

After looking at the pictures, we found that some of them labeld as "falling" but contained multiple people who some of them were not actually falling (but only one of them) . Therefore, in order to improve our data, we created a tool that go through all of the data and extracted every person from each photo into a different photo. After that, we once again cleaned and tagged all of the new data.

Unfortunately, the model did not succeed in learning on the new data either.
The amount of the data, which contains approximately 500 photos, is likely the key factor in the network learning failure, but this is the only data that is available on the Internet.

The original image database can be found [here](https://github.com/bakshtb/Human-Fall-Detection/tree/master/fall_dataset/old "here"), while the new image database we made can be found [here](https://github.com/bakshtb/Human-Fall-Detection/tree/master/fall_dataset/images "here").


We considered creating a straightforward if-else-based model that would detect a fall in accordance with logical conditions regarding the relative location of the body parts after the LSTM-based network failed to learn from our image library.

## Results

Using our own data, we tested our model, and the results were, in our opinion, pretty good, as follows:

| Accuracy  | Precision | Recall | F1 score |
| ------------- | ------------- | ------------- | ------------- |
|  81.18%  | 83.27% | 83.58%  | 83.42%  |

For the video analysis we used Nvidia's Tesla K80 GPU, the system analyzes the videos at a reasonable speed of 15Fps.

## How To Use
- Clone this repository into your drive
- Download the [YOLOv7-POSE](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt "YOLOv7-POSE") model into `Human-Fall-Detection` directory.
- Install all the requirements with `pip install -r requirements.txt`
- `video.py`: Run fall detection on videos file
- `realtime.py`: Run real-time fall detection via webcam


## Examples
Examples of videos collected from the Internet and analyzed by our system are shown below.

These videos demonstrate how the model successfully and accurately recognizes human falls.

![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_1_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_2_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_4_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_5_keypoint.gif)
![](https://github.com/bakshtb/Human-Fall-Detection/blob/master/fall_dataset/results/video_6_keypoint.gif)

## Work in Progress
- Currently implementing real-time fall detection logic
- **Short Term**: Optimize the if-else-based model using a time-sliding window
- **Long Term**: Integrate a time-series model (e.g., LSTM) for more accurate detection

## Fall Detect Logic
The fall detection logic uses the following parameters:

| Parameter              | Description |
|------------------------|-------------|
| `FPS`                 | Frames per second. Used to calculate time and velocity. |
| `WINDOW_SIZE`         | Number of frames in each analysis window. Defines how many frames are used to evaluate pose changes. |
| `V_THRESH`            | Threshold for center-of-mass velocity. Movements faster than this are considered potentially abnormal. |
| `DY_THRESH`           | Threshold for vertical (Y-axis) displacement of the center of mass. |
| `ASPECT_RATIO_THRESH` | Threshold for change in aspect ratio (width/height) of the body. Indicates whether the body has become horizontal. |

These values can be configured via a `.env` file.  
If not provided, default values defined in the code will be used.

> **Note:** This logic is still under development and subject to change as part of our ongoing implementation and validation process.


## Possible Future Improvements

In order to alert human falls and save lives, the real-time system may be deployed and implemented in nursing homes, hospitals, and senior living facilities.

