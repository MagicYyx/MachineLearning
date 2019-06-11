#!/usr/bin/env python
# -*- coding:utf-8 -*-
from KNN.KNN_Classifier import KNN_Classifier


path = r'C:\Users\Administrator\Desktop\data.xlsx'
x = [[0.1, 0.3, 0.3] ,[0.3, 0.2, 0.1]]
knn = KNN_Classifier()
knn.fit(path,True,False)
print(knn.predict(x))
