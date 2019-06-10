# -*- coding: utf-8 -*-

"""
KNN
"""

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, metrics


# 获取数据，并拆分为训练集和测试集
def get_data(path):
    data = pd.read_excel(path)
    predictors = data.columns[:-1]
    label = data.columns[-1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data[predictors], data[label], test_size=0.25)
    counts = x_train.shape[0]
    return x_train, x_test, y_train, y_test, counts


# 交叉验证法得出最佳K
def k_test(path):
    x_train, x_test, y_train, y_test, counts = get_data(path)
    # 设置k值集合
    K = np.arange(1, int(np.ceil(np.log2(counts))))
    # 存储不同k值的平均准确率
    accuracy = []
    for k in K:
        # 使用10折交叉验证的方法，比对每一k值下KNN模型的预测准确率
        cv_result = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'),
                                                    x_train, y_train, cv=10, scoring='accuracy')
        accuracy.append(cv_result.mean())

    # 查询最大平均准确率的下标
    arg_max = np.array(accuracy).argmax()
    # 绘制折线图
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文和负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(K, accuracy)  # 折线图
    plt.scatter(K, accuracy)  # 散点图
    plt.text(K[arg_max], accuracy[arg_max], '最佳k值为%s' % int(K[arg_max]))
    plt.show()  # 显示图形
    return int(K[arg_max])


# 以最佳K值构建模型
def train_model(path):
    k_best = k_test(path)
    knn_class = neighbors.KNeighborsClassifier(n_neighbors=k_best, weights='distance')
    knn_class.fit(x_train, y_train)
    predict = knn_class.predict(x_test)




# 模型评估
print('Confusion Matrix:\n', pd.crosstab(y_test, predict), '\n')  # 构建混淆矩阵
print('Overall Accuracy:', metrics.scorer.accuracy_score(y_test, predict), '\n')  # 整体准确率
print('Assessment Report:\n', metrics.classification_report(y_test, predict))  # 模型评估报告
