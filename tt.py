#!/usr/bin/env python
# -*- coding:utf-8 -*-
from KNN.KNN import KNN
from sklearn.preprocessing import minmax_scale
import pandas as pd

path = r'C:\Users\Administrator\Desktop\data.xlsx'
x = [[40920, 8.326976, 0.953952]]
knn = KNN()
knn.fit(path, 'Classifier', normalization=True, show_picture=False)
print(knn.predict(x))