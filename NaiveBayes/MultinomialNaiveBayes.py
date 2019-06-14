# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn import model_selection,naive_bayes,metrics
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\Administrator\Desktop\mushrooms.csv')
#将字符型数据做因子化处理，将其转换为整数型数据
columns=data.columns[1:]
for column in columns:
    data[column]=pd.factorize(data[column])[0]

#拆分为训练集和测试集
x_train,x_test,y_train,y_test=model_selection.train_test_split(data[columns],data.type,
                                                               test_size=0.25,random_state=1234)
#调用多项式朴素贝叶斯
mnb=naive_bayes.MultinomialNB()
mnb.fit(x_train,y_train)
mnb_pred=mnb.predict(x_test)
#显示预测结果，各类别的预测数量

#模型检验
print('模型的准确率为：',metrics.accuracy_score(y_test,mnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test,mnb_pred))
#绘制ROC曲线
y_score=mnb.predict_proba(x_test)[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test.map({'edible':0,'poisonous':1}),y_score)
roc_auc=metrics.auc(fpr,tpr)
plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
plt.plot(fpr,tpr,color='black',lw=1)
plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.text(0.5,0.3,'ROC Curve (area=%0.2f)' % roc_auc)
plt.xlabel('l-Specificity')
plt.ylabel('Sensitivity')
plt.show()