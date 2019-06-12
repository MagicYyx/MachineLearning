#!/usr/bin/env python
# -*- coding:utf-8 -*-
from DecisionTree.DecisionTree import DecisionTree

path = r'C:\Users\Administrator\Desktop\myGit\MachineLearning\DecisionTree\NHANES.xlsx'
max_depth = [20, 21, 22, 23, 24]
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [2, 4, 6, 8, 10]
dt = DecisionTree()
dt.fit(path, 'Regressor', max_depth, min_samples_split, min_samples_leaf)
