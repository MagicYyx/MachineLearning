# -*- coding: utf-8 -*-

"""
KNN
"""

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, neighbors, metrics
from sklearn.preprocessing import minmax_scale


class KNN_Classifier(object):
    # 获取数据
    def __get_data(self, path):
        self.__data = pd.read_excel(path)
        self.__counts = self.__data.shape[0]

    # 将数据拆分为训练集和测试集
    def __split_data(self, normalization=False):
        predictors = self.__data.columns[:-1]
        label = self.__data.columns[-1]
        if normalization:
            X = minmax_scale(self.__data[predictors])
        else:
            X = self.__data[predictors]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = model_selection.train_test_split(
            X, self.__data[label],
            test_size=0.25)
        print(self.__x_test)

    # 交叉验证法得出最佳K
    def __k_test(self, show_picture):
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
        if show_picture:
            # 绘制折线图
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文和负号正常显示
            plt.rcParams['axes.unicode_minus'] = False
            plt.plot(K, accuracy)  # 折线图
            plt.scatter(K, accuracy)  # 散点图
            plt.text(K[arg_max], accuracy[arg_max], '最佳k值为%s' % int(K[arg_max]))
            plt.show()  # 显示图形
        else:
            print('最佳k值为%s' % K[arg_max])
        return int(K[arg_max])

    # 以最佳K值构建模型
    def __train_model(self, show_picture):
        k_best = self.__k_test(show_picture)
        self.__knn_class = neighbors.KNeighborsClassifier(n_neighbors=k_best, weights='distance')
        self.__knn_class.fit(self.__x_train, self.__y_train)
        self.__predict = self.__knn_class.predict(self.__x_test)

    # 模型评估
    def __model_evaluation(self):
        print('Confusion Matrix:\n', pd.crosstab(self.__y_test, self.__predict), '\n')  # 构建混淆矩阵
        print('Overall Accuracy:', metrics.scorer.accuracy_score(self.__y_test, self.__predict), '\n')  # 整体准确率
        print('Assessment Report:\n', metrics.classification_report(self.__y_test, self.__predict))  # 模型评估报告

    # 执行全部流程
    def fit(self, path, normalization=False, show_picture=True):
        self.__get_data(path)
        self.__split_data(normalization)
        self.__train_model(show_picture)
        self.__model_evaluation()

    # 预测函数
    def predict(self, data):
        '''
        :param data: 如[[1.3, 2.2]]、[[1,2], [2,3]]的形式
        :return:
        '''
        return self.__knn_class.predict(data)
