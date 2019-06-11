#!/usr/bin/env python
# -*- coding:utf-8 -*-
from KNN.KNN_Classifier import KNN_Classifier


path = r'C:\Users\Administrator\Desktop\data.xlsx'
x = [0.1, 0.3, 0.3]
knn = KNN_Classifier(path)
knn.fit()
print(knn.predict(x))
