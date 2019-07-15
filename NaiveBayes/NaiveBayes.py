#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高斯贝叶斯分类器
"""

__author__ = 'Magic Yyx'

import pandas as pd
from sklearn import model_selection, naive_bayes, metrics
from sklearn.externals import joblib
# import matplotlib.pyplot as plt


class Bayes(object):
    # 拆分为训练集和测试集
    def __split_data(self, data):
        x = data.columns[:-1]
        y = data.columns[-1]
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = model_selection.train_test_split(data[x],
                                                                                                        data[y],
                                                                                                        test_size=0.25)

    # 训练模型
    def __train_model(self, type):
        if type == 'Gaussian':
            self.__bayes_model = naive_bayes.GaussianNB()
        elif type == 'Multinomial':
            self.__bayes_model = naive_bayes.MultinomialNB()
        elif type == 'Bernoulli':
            self.__bayes_model = naive_bayes.BernoulliNB()
        self.__bayes_model.fit(self.__x_train, self.__y_train)

    # 模型评估
    def __model_evaluation(self):
        print('模型的准确率为：', metrics.accuracy_score(self.__y_test, self.__bayes_model.predict(self.__x_test)))
        print('模型的评估报告：\n', metrics.classification_report(self.__y_test, self.__bayes_model.predict(self.__x_test)))

    # 执行全部流程
    def fit(self, data, type):
        self.__split_data(data)
        self.__train_model(type)
        self.__model_evaluation()

    # 模型预测
    def predict(self):
        pass

    # 保存模型
    def save_model(self, path):
        joblib.dump(self.__bayes_model, path)


# # 设置正例和负例，便于后面画ROC曲线
# data.y = data.y.map({2: 0, 1: 1})
# # 绘制ROC曲线
# y_score = gnb.predict_proba(x_test)[:, 1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, y_score)
# roc_auc = metrics.auc(fpr, tpr)
# plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
# plt.plot(fpr, tpr, color='black', lw=1)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.text(0.5, 0.3, 'ROC Curve (area=%0.2f)' % roc_auc)
# plt.xlabel('l-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()
