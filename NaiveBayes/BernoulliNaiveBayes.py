# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn import model_selection,naive_bayes,metrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import jieba

data=pd.read_excel(r'C:\Users\Administrator\Desktop\Contents.xlsx')
#数据清洗
data.Content=data.Content.str.replace('[0-9a-zA-Z]','')

#加载自定义词库
jieba.load_userdict(r'C:\Users\Administrator\Desktop\all_words.txt')
#读入停用词
with open(r'C:\Users\Administrator\Desktop\mystopwords.txt',encoding='utf-8') as words:
    stop_words=[i.strip() for i in words.readlines()]
#构造切词的自定义函数，在切词过程中删除停用词
def cut_word(centence):
    words=[i for i in jieba.lcut(centence) if i not in stop_words]
    result=' '.join(words)
    return(result)
#调用函数切词
words=data.Content.apply(cut_word)

#计算每个词出现的次数，并将稀疏度99%以上的词删除
counts=CountVectorizer(min_df=0.01)
#文档词条矩阵
dtm_counts=counts.fit_transform(words).toarray()
#文档的列名称
columns=counts.get_feature_names()
#形成x变量,y变量
x=pd.DataFrame(dtm_counts,columns=columns)
y=data.Type

#拆分为训练集和测试集
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.25,random_state=1234)
#调用伯努利朴素贝叶斯
bnb=naive_bayes.BernoulliNB()
bnb.fit(x_train,y_train)
bnb_pred=bnb.predict(x_test)
#显示预测结果，各类别的预测数量

#模型检验
print('模型的准确率为：',metrics.accuracy_score(y_test,bnb_pred))
print('模型的评估报告：\n',metrics.classification_report(y_test,bnb_pred))
#绘制ROC曲线
y_score=bnb.predict_proba(x_test)[:,1]
fpr,tpr,threshold=metrics.roc_curve(y_test.map({'Negative':0,'Positive':1}),y_score)
roc_auc=metrics.auc(fpr,tpr)
plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
plt.plot(fpr,tpr,color='black',lw=1)
plt.plot([0,1],[0,1],color='red',linestyle='--')
plt.text(0.5,0.3,'ROC Curve (area=%0.2f)' % roc_auc)
plt.xlabel('l-Specificity')
plt.ylabel('Sensitivity')
plt.show()