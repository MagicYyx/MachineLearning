#!/usr/bin/env python
# -*- coding:utf-8 -*-
from RandomForest.RandomForest import RandomForest

path = r'C:\Users\Administrator\Desktop\myGit\MachineLearning\DecisionTree\Titanic.xlsx'
dt = RandomForest()
dt.fit(path, 'Classifier')
