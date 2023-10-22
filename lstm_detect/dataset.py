import csv
import numpy as np
import torch
import pandas as pd
import numpy as np
from datetime import datetime

TIME_STEPS = 80


def create_sequences(X, time_steps=TIME_STEPS):
    if len(X) < time_steps:
        return [], []
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X[i:(i+time_steps)])
        ys.append(X[i+time_steps])

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


class DetectDataSet(object):
    def __init__(self, data_path):
        # 读取CSV文件
        data_frame = pd.read_csv(data_path)

        # 提取关键元素的列
        key_elements = ['TICKS', 'CPU%', 'Memory_per%', 'Net_recv', 'Net_sent']
        data = data_frame[key_elements].values

        # 转换时间戳为时间格式
        timestamps = data[:, 0]  # 提取TICKS列
        timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]

        # 设置时间步数
        TIME_STEPS = 80
        tensor = []
        for i in range(len(data) - TIME_STEPS + 1):
            sequence = data[i:i + TIME_STEPS, 1:]  # 提取除了时间戳之外的其他元素
            tensor.append(sequence)

        tensor = np.array(tensor)


    def __getitem__(self, index):
        return self.data['input'][index], self.data['label'][index], self.data['timestamp'][index]

    def __len__(self):
        return self.length
