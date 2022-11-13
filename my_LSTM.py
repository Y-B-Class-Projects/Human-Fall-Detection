import os

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


def get_data_from_images(folder_path):
    x = []
    y = []
    for folder in os.listdir(folder_path):
        for file in tqdm(os.listdir(folder_path + '/' + folder)[:3]):
            pose = image_to_pose(folder_path + '/' + folder + '/' + file)
            if len(pose) >= 1:
                pose = pose[0][7:]
                _pose = []
                for i in range(17):  # 17 keypoints
                    _x = pose[3 * i]
                    _y = pose[3 * i + 1]
                    _pose.append(_x)
                    _pose.append(_y)
                x.append(_pose)
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

    x_train = [list(x) for x in x_train]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_train = [list(x) for x in x_train]

    # split into train and test sets
    train_size = int(len(dataset) * 0.80)
    x_train, x_test = x_train[0:train_size], x_train[train_size:len(dataset)]
    y_train, y_test = y_train[0:train_size], y_train[train_size:len(dataset)]

    x_train = np.expand_dims(x_train, axis=0)
    y_train = np.array(list(y_train))
    y_train = y_train.reshape(1, -1)

    x_test = [[x] for x in x_test]

    model = Sequential()
    model.add(LSTM(16, return_sequences=False, input_shape=(None, x_train.shape[2])))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))
    model.fit(x=x_train, y=y_train)

    y_predict = model.predict(x_test)
    # for y, y_pred in zip(y_test, y_predict):
    #     print('y', str(y), 'y_pred', str(int(y_pred)))

    # accuracy: (tp + tn) / (p + n)
    y_test = list(y_test)
    y_predict = [int(y) for y in y_predict]

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


# train()
clean_images('fall_dataset/images/fall', 'fall_dataset/images/cropped_images')
clean_images('fall_dataset/images/not-fall', 'fall_dataset/images/cropped_images')
