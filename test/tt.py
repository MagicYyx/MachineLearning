#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pymongo
import re
import cpca
import pandas as pd

myClient = pymongo.MongoClient('localhost')
myDB = myClient['weibo']
myCol = myDB['中国地震台网速报']

x = myCol.find()
pattern = re.compile('.*正式测定.*(\d{2}月\d{2}日\d{2}时\d{2}分)在(.*?)(（.*）)发生(.*?)级.*?地震.*震源深度(.*?)千米')
tmp_list = []
for i in range(2775): #2775, 2849
    lst = pattern.findall(x[i]['txt'])
    if len(lst) != 0:
        print(x[i]['time'], ' ', lst)
        tmp_list.append(lst[0][1])

df = cpca.transform(tmp_list)
print(df['区'].value_counts())