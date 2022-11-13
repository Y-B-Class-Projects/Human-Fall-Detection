import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from poseEstimation import image_to_pose

FALL = 1
NOT_FALL = 0


def get_data_from_images(folder_path):
    x = []
    y = []
    for folder in os.listdir(folder_path):
        for file in os.listdir(folder_path + '/' + folder)[:3]:
            x.append(image_to_pose(folder_path + '/' + folder + '/' + file)[0])
            y.append(FALL if folder == 'fall' else NOT_FALL)
    return pd.DataFrame({'x': x, 'y': y})


def store_data(df, folder_path):
    df.to_csv(folder_path + '/data.csv', index=False)
    df.to_pickle(folder_path + '/data.pkl')


def load_data(folder_path):
    return pd.read_pickle(folder_path + '/data.pkl')


def train(from_images=True):
    if from_images:
        data = get_data_from_images('fall_dataset/images')
        store_data(data, 'fall_dataset/data')
    else:
        data = load_data('fall_dataset/data')

    dataset = data.values

    print(dataset)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # dataset = scaler.fit_transform(dataset)


train()
