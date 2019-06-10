#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getDataSet(file_path):
    data = pd.read_excel(file_path)
    # 标签放在最后一列
    x_heads = data.columns[:-1]
    y_heads = data.columns[-1]
    x = np.array(data[x_heads])
    y = np.array(data[y_heads])
    return x, y


def classify(classify_item, k, file_path):
    dataSet, labels = getDataSet(file_path)
    # 计算距离
    distance = np.sum((classify_item - dataSet) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [labels[index] for index in distance.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    result = max(k_labels, key=k_labels.count)
    return result


def picture(file_path):
    x, y = getDataSet(file_path)
    LabelColors = []
    for i in range(len(y)):
        if y[i] == 1:
            LabelColors.append('black')
        if y[i] == 2:
            LabelColors.append('red')
        if y[i] == 3:
            LabelColors.append('blue')
        if y[i] == 4:
            LabelColors.append('orange')
    plt.scatter(x[:,0], x[:,1],color=LabelColors)
    plt.show()
