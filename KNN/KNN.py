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


class KNN(object):
    # 获取数据
    def __get_data(self, path):
        self.__data = pd.read_excel(path)
        self.__counts = self.__data.shape[0]

    # 将数据拆分为训练集和测试集
    def __split_data(self, normalization=False):
        predictors = self.__data.columns[:-1]
        label = self.__data.columns[-1]
        if normalization:
            X = minmax_scale(self.__data[predictors])  # 数据归一化，消除量纲的影响
        else:
            X = self.__data[predictors]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = model_selection.train_test_split(
            X, self.__data[label],
            test_size=0.25)

    # 交叉验证法得出最佳K
    def __k_test(self, show_picture):
        # 设置k值集合
        K = np.arange(1, int(np.ceil(np.log2(self.__counts))))
        # 存储不同k值下的准确率或者均方误差
        measure = []
        if self.__type == 'Classifier':
            for k in K:
                # 使用10折交叉验证的方法，比对每一k值下KNN模型的预测准确率
                cv_result = model_selection.cross_val_score(
                    neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'),
                    self.__x_train, self.__y_train, cv=10, scoring='accuracy')
                measure.append(cv_result.mean())
            # 查询最大平均准确率的下标
            K_index = np.array(measure).argmax()
        elif self.__type == 'Regressor':
            for k in K:
                # 使用10折交叉验证的方法，比对每一k值下KNN模型的MSE
                cv_result = model_selection.cross_val_score(
                    neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance'),
                    self.__x_train, self.__y_train, cv=10, scoring='neg_mean_squared_error')
                measure.append((-1 * cv_result).mean())  # 将负数转换为正数
            # 查询最小均方误差的下标
            K_index = np.array(measure).argmin()
        if show_picture:
            # 绘制折线图
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文和负号正常显示
            plt.rcParams['axes.unicode_minus'] = False
            plt.plot(K, measure)  # 折线图
            plt.scatter(K, measure)  # 散点图
            plt.text(K[K_index], measure[K_index], '最佳k值为%s' % K[K_index])
            plt.show()  # 显示图形
        else:
            print('最佳k值为%s' % K[K_index])
        return int(K[K_index])

    # 以最佳K值训练模型
    def __train_model(self, show_picture):
        k_best = self.__k_test(show_picture)
        if self.__type == 'Classifier':
            self.__knn = neighbors.KNeighborsClassifier(n_neighbors=k_best, weights='distance')
        elif self.__type == 'Regressor':
            self.__knn = neighbors.KNeighborsRegressor(n_neighbors=k_best, weights='distance')
        self.__knn.fit(self.__x_train, self.__y_train)
        self.__predict = self.__knn.predict(self.__x_test)

    # 模型评估
    def __model_evaluation(self):
        if self.__type == 'Classifier':
            print('Confusion Matrix:\n', pd.crosstab(self.__y_test, self.__predict), '\n')  # 构建混淆矩阵
            print('Overall Accuracy:', metrics.scorer.accuracy_score(self.__y_test, self.__predict), '\n')  # 整体准确率
            print('Assessment Report:\n', metrics.classification_report(self.__y_test, self.__predict))  # 模型评估报告
        elif self.__type == 'Regressor':
            print('MSE:', metrics.mean_squared_error(self.__y_test, self.__predict))  # 均方误差越小越好

    # 执行全部流程
    def fit(self, path, type, normalization=False, show_picture=True):
        '''
        :param path: 数据文件路径
        :param type: 分类或回归（"Classifier" or "Regressor"）
        :param normalization: 数据是否归一化
        :param show_picture: 是否显示寻找最佳k值的图像
        :return:
        '''
        self.__type = type
        self.__normalization = normalization
        # 获取数据
        self.__get_data(path)
        # 拆分训练集和测试集
        self.__split_data(normalization)
        # 训练模型
        self.__train_model(show_picture)
        # 模型评估
        self.__model_evaluation()

    # 预测函数
    def predict(self, data):
        '''
        :param data: 如[[1.3, 2.2]]、[[1,2], [2,3]]的形式
        :return:
        '''
        if self.__normalization:  # 做了归一化的，预测数据也要做归一化处理
            maxs = self.__data[self.__data.columns[:-1]].max()
            mins = self.__data[self.__data.columns[:-1]].min()
            devisions = maxs - mins
            for outer_index in range(len(data)):
                for inner_index in range(len(data[0])):
                    data[outer_index][inner_index] = data[outer_index][inner_index] / devisions[inner_index]
        return self.__knn.predict(data)
