# -*- coding: utf-8 -*-

"""
RandomForest
正例和负例请分别用1和0表示，不要用其他的
"""

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
from sklearn import model_selection, ensemble, metrics
import matplotlib.pyplot as plt


class RandomForest(object):
    # 获取数据
    def __get_data(self, path):
        self.__data = pd.read_excel(path)

    # 拆分为训练集和测试集
    def __split_data(self):
        x = self.__data.columns[:-1]
        y = self.__data.columns[-1]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = model_selection.train_test_split(
            self.__data[x], self.__data[y],
            test_size=0.25)

    # 构建随机森林
    def __train_model(self):
        if self.__type == 'Classifier':
            self.__random_forest = ensemble.RandomForestClassifier(n_estimators=200)
        elif self.__type == 'Regressor':
            self.__random_forest = ensemble.RandomForestRegressor(n_estimators=200)
        self.__random_forest.fit(self.__x_train, self.__y_train)

    # 模型评估
    def __model_evaluation(self):
        if self.__type == 'Classifier':
            print('训练集的预测准确率:', metrics.accuracy_score(self.__y_train, self.__random_forest.predict(self.__x_train)))
            print('测试集的预测准确率:', metrics.accuracy_score(self.__y_test, self.__random_forest.predict(self.__x_test)))
            # 绘制ROC曲线
            y_score = self.__random_forest.predict_proba(self.__x_test)[:, 1]  # 预测值为第2种的概率
            fpr, tpr, threshold = metrics.roc_curve(self.__y_test, y_score)  # 正例和负例用1和0表示
            # 计算AUC值
            auc_area = metrics.auc(fpr, tpr)
            # 绘制面积图
            plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
            # 绘制曲线图
            plt.plot(fpr, tpr, color='black', lw=1)
            # 对角线
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            # 添加文本信息
            plt.text(0.4, 0.4, 'DecisionTree ROC curve (area = %0.2f)' % auc_area)
            # 添加x轴和y轴标签
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.show()
        elif self.__type == 'Regressor':
            print('MSE:', metrics.mean_squared_error(self.__y_test,self.__random_forest.predict(self.__x_test)))

    # 执行全部流程
    def fit(self, path, type):
        self.__type = type
        self.__get_data(path)
        self.__split_data()
        self.__train_model()
        self.__model_evaluation()
