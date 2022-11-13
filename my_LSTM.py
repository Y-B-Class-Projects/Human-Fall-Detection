import math
import os
import urllib

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from PIL import Image
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense
from matplotlib import patches
from pytorch_lightning.loggers import wandb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from poseEstimation import image_to_pose
from tqdm import tqdm

FALL = 1
NOT_FALL = 0


def fall_detection(pose):
    xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
    xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
    left_shoulder_y = pose[23]
    left_shoulder_x = pose[22]
    right_shoulder_y = pose[26]
    left_body_y = pose[41]
    left_body_x = pose[40]
    right_body_y = pose[44]
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
    left_foot_y = pose[53]
    right_foot_y = pose[56]
    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
            len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2):
        return True
    return False


def get_data_from_images(folder_path):
    x = []
    y = []
    for folder in os.listdir(folder_path):
        for file in tqdm(os.listdir(folder_path + '/' + folder)[:3]):
            pose = image_to_pose(folder_path + '/' + folder + '/' + file)
            if len(pose) >= 1:
                pose = pose[0]
                x.append(pose)
                y.append(FALL if folder == 'fall' else NOT_FALL)
    return pd.DataFrame({'x': x, 'y': y})


def store_data(df, folder_path):
    df.to_csv(folder_path + '/data.csv', index=False)
    df.to_pickle(folder_path + '/data.pkl')


def load_data(folder_path):
    return pd.read_pickle(folder_path + '/data.pkl')


def train(from_images=False):
    if from_images:
        data = get_data_from_images('fall_dataset/images')
        store_data(data, 'fall_dataset/data')
    else:
        data = load_data('fall_dataset/data')

    dataset = data.values

    np.random.shuffle(dataset)

    x_train = dataset[:, 0]
    y_train = dataset[:, 1]

    # split into train and test sets
    train_size = int(len(dataset) * 0.60)
    x_train, x_test = x_train[0:train_size], x_train[train_size:len(dataset)]
    y_train, y_test = y_train[0:train_size], y_train[train_size:len(dataset)]

    # y_predict = ...
    # for y, y_pred in zip(y_test, y_predict):
    #     print('y', str(y), 'y_pred', str(int(y_pred)))

    # accuracy: (tp + tn) / (p + n)
    y_test = list(y_test)
    y_predict = [int(y) for y in y_predict]

    print('y_predict', y_predict)

    accuracy = accuracy_score(y_test, y_predict)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_predict)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_predict)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_predict)
    print('F1 score: %f' % f1)


def crop_humans_from_image(image_path, save_path='fall_dataset/images/cropped_images'):
    img = Image.open(image_path)
    img_name = '-'.join(image_path.split('/')[-1].split('.')[:-1])
    img_type = image_path.split('/')[-1].split('.')[-1]
    w, h = img.size
    new_w, new_h = 960, int((960 / w) * h)

    pose = image_to_pose(image_path)
    for idx in range(len(pose)):
        xmin, ymin = (pose[idx, 2] - pose[idx, 4] / 2), (pose[idx, 3] - pose[idx, 5] / 2)
        xmax, ymax = (pose[idx, 2] + pose[idx, 4] / 2), (pose[idx, 3] + pose[idx, 5] / 2)
        xmin, ymin = int(xmin * w / new_w), int(ymin * h / new_h)
        xmax, ymax = int(xmax * w / new_w), int(ymax * h / new_h)
        xmin, ymin = xmin - 5, ymin - 5
        xmax, ymax = xmax + 5, ymax + 5
        crop = img.crop((xmin, ymin, xmax, ymax))
        crop.save(save_path + '/' + img_name + '_cropped_' + str(idx) + '.' + img_type)


def clean_images(folder_path, save_path):
    for file in tqdm(os.listdir(folder_path)):
        crop_humans_from_image(folder_path + '/' + file)


def predict(image_url):
    urllib.request.urlretrieve(image_url, "image.png")
    pose = image_to_pose("image.png")
    if len(pose) >= 1:
        pose = pose[0][7:]
        _pose = []
        for i in range(17):  # 17 keypoints
            _x = pose[3 * i]
            _y = pose[3 * i + 1]
            _pose.append(_x)
            _pose.append(_y)
        x = np.array(_pose)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x[:, np.newaxis])
        x = [[[_x[0] for _x in x]]]
        model = tf.keras.models.load_model('my_model')
        y_predict = model.predict(x)
        print('y_predict', str(int(y_predict)))
        return int(y_predict[0][0])
    return -1


# predict('https://media.istockphoto.com/id/1307214736/photo/full-length-portrait-of-a-corpulent-mature-man-posing.jpg?b=1&s=170667a&w=0&k=20&c=7BWU__SfBwYmX9coO-_cy-mCrE0RTDXWIku9Jikggcc=')
train(True)
# clean_images('fall_dataset/images/fall', 'fall_dataset/images/cropped_images')
# clean_images('fall_dataset/images/not-fall', 'fall_dataset/images/cropped_images')
