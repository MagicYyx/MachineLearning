# -*- coding: utf-8 -*-

"""
KNN
"""

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, metrics


class KNN_Classifier(object):
    def __init__(self, path):
        self.__path = path
        self.__data = pd.read_excel(path)
        self.__counts = self.__data.shape[0]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = self.__split_data()

    # 将数据拆分为训练集和测试集
    def __split_data(self):
        predictors = self.__data.columns[:-1]
        label = self.__data.columns[-1]
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.__data[predictors], self.__data[label],
                                                                            test_size=0.25)
        return x_train, x_test, y_train, y_test

    # 交叉验证法得出最佳K
    def __k_test(self):
        # 设置k值集合
        K = np.arange(1, int(np.ceil(np.log2(self.__counts))))
        # 存储不同k值的平均准确率
        accuracy = []
        for k in K:
            # 使用10折交叉验证的方法，比对每一k值下KNN模型的预测准确率
            cv_result = model_selection.cross_val_score(
                neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'),
                self.__x_train, self.__y_train, cv=10, scoring='accuracy')
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
    def __train_model(self):
        k_best = self.__k_test()
        self.__knn_class = neighbors.KNeighborsClassifier(n_neighbors=k_best, weights='distance')
        self.__knn_class.fit(self.__x_train, self.__y_train)
        self.__predict = self.__knn_class.predict(self.__x_test)

    # 预测函数
    def predict(self, data):
        data_new = [data]
        return self.__knn_class.predict(data_new)

    # 模型评估
    def __model_evaluation(self):
        print('Confusion Matrix:\n', pd.crosstab(self.__y_test, self.__predict), '\n')  # 构建混淆矩阵
        print('Overall Accuracy:', metrics.scorer.accuracy_score(self.__y_test, self.__predict), '\n')  # 整体准确率
        print('Assessment Report:\n', metrics.classification_report(self.__y_test, self.__predict))  # 模型评估报告

    # 执行全部流程
    def fit(self):
        self.__train_model()
        self.__model_evaluation()
