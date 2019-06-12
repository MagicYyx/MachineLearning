# -*- coding: utf-8 -*-

"""
DecisionTree
正例和负例请分别用1和0表示，不要用其他的
"""

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
from sklearn import model_selection, tree, metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


class DecisionTree(object):
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

    # 使用网格搜索法，寻找最优参数组合（最大深度、分支最小样本量、叶节点最小样本量）
    # 根据经验，数据量较小时树的最大深度可设置10以内，较大时则需设置比较大的树深度，如20左右
    def __find_best_params(self, max_depth, min_samples_split, min_samples_leaf):
        params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
        if self.__type == 'Classifier':
            grid = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=params, cv=10)
        elif self.__type == 'Regressor':
            grid = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid=params, cv=10)
        grid.fit(self.__x_train, self.__y_train)
        print('best params:\n', grid.best_params_)
        return grid.best_params_

    # 构建决策树
    def __train_model(self, max_depth, min_samples_split, min_samples_leaf):
        best_params = self.__find_best_params(max_depth, min_samples_split, min_samples_leaf)
        if self.__type == 'Classifier':
            self.__decision_tree = tree.DecisionTreeClassifier(max_depth=best_params["max_depth"],
                                                               min_samples_leaf=best_params['min_samples_leaf'],
                                                               min_samples_split=best_params['min_samples_split'])
        elif self.__type == 'Regressor':
            self.__decision_tree = tree.DecisionTreeRegressor(max_depth=best_params["max_depth"],
                                                              min_samples_leaf=best_params['min_samples_leaf'],
                                                              min_samples_split=best_params['min_samples_split'])
        self.__decision_tree.fit(self.__x_train, self.__y_train)

    # 模型评估
    def __model_evaluation(self):
        if self.__type == 'Classifier':
            print('训练集的预测准确率:', metrics.accuracy_score(self.__y_train, self.__decision_tree.predict(self.__x_train)))
            print('测试集的预测准确率:', metrics.accuracy_score(self.__y_test, self.__decision_tree.predict(self.__x_test)))
            # 绘制ROC曲线
            y_score = self.__decision_tree.predict_proba(self.__x_test)[:, 1]  # 预测值为第2种的概率
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
            print('MSE:', metrics.mean_squared_error(self.__y_test,self.__decision_tree.predict(self.__x_test)))

    # 执行全部流程
    def fit(self, path, type, max_depth, min_samples_split, min_samples_leaf):
        self.__type = type
        self.__get_data(path)
        self.__split_data()
        self.__train_model(max_depth, min_samples_split, min_samples_leaf)
        self.__model_evaluation()
