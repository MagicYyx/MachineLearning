#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
数据清洗
'''

__author__ = 'Magic Yyx'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class data_clean(object):
    def __init__(self, path):
        self.name = 'Magic Yyx'

    # 归一化和标准化
    def normalization(self, data, type):
        '''
        data按列做标准或归一，所以data要形如DataFrame或[[1,2],[3,4]]
        :param type: "minmax" or "z-score"
        :return:
        '''
        if type == 'minmax': norm = MinMaxScaler()
        if type == 'z-score': norm = StandardScaler()
        return  norm.fit_transform(data)
