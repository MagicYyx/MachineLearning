#!/usr/bin/env python
# -*- coding:utf-8 -*-
from RandomForest.RandomForest import RandomForest
from DecisionTree.DecisionTree import DecisionTree

path = r'C:\Users\Administrator\Desktop\myGit\MachineLearning\DecisionTree\Titanic.xlsx'
x = [[3, 1, 1, 7.2]]
dt = DecisionTree()
rf = RandomForest()
max_depth=[2,3,4,5,6]
min_samples_split=[2,4,6,8]
min_samples_leaf=[2,4,8,10,12]
dt.fit(path, 'Classifier', max_depth, min_samples_split, min_samples_leaf)
print(dt.predict(x))
rf.fit(path, 'Classifier')
print(rf.predict(x))
