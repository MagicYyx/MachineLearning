#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    # labels = ['1', '2', '3', '4']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # 计算距离
    distance = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [labels[index] for index in distance.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    result = max(k_labels, key=k_labels.count)
    return result

if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([1,3], group, labels, 3))
