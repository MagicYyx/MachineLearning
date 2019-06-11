#!/usr/bin/env python
# -*- coding:utf-8 -*-
from KNN.KNN_Classifier import KNN_Classifier


path = r'C:\Users\Administrator\Desktop\myGit\MachineLearning\KNN\Knowledge.xlsx'
x = [0.1, 0.1, 0.15, 0.66, 0.33]
knn = KNN_Classifier(path)
knn.fit()
print(knn.predict(x))
