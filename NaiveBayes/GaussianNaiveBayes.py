#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高斯贝叶斯分类器
"""

__author__ = 'Magic Yyx'

import pandas as pd
from sklearn import model_selection,naive_bayes,metrics
import matplotlib.pyplot as plt

data=pd.read_excel(r'C:\Users\Administrator\Desktop\Skin_Segment.xlsx')
#拆分为训练集和测试集
#设置正例和负例，便于后面画ROC曲线
data.y=data.y.map({2:0,1:1})
x_train,x_test,y_train,y_test=model_selection.train_test_split(data.iloc[:,:3],data.y,
                                                               test_size=0.25,random_state=1234)
#调用高斯朴素贝叶斯
gnb=naive_bayes.GaussianNB()
gnb.fit(x_train,y_train)
gnb_pred=gnb.predict(x_test)
#显示预测结果，各类别的预测数量
print('prediction:\n',pd.Series(gnb_pred).value_counts())

#模型检验
print('模型的准确率为：',metrics.accuracy_score(y_test,gnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test,gnb_pred))
#绘制ROC曲线
y_score=gnb.predict_proba(x_test)[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test,y_score)
roc_auc=metrics.auc(fpr,tpr)
plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
plt.plot(fpr,tpr,color='black',lw=1)
plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.text(0.5,0.3,'ROC Curve (area=%0.2f)' % roc_auc)
plt.xlabel('l-Specificity')
plt.ylabel('Sensitivity')
plt.show()