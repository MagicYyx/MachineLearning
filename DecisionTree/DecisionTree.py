# -*- coding: utf-8 -*-
"""
tmp code
"""
import pandas as pd
import numpy as np
from sklearn import model_selection

class DecisionTree(object):
    # 获取数据
    def __get_data(self, path):
        self.__data = pd.read_excel(path)

    # 拆分为训练集和测试集
    def __split_data(self):
        pass