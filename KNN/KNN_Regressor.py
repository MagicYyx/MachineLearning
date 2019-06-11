# -*- coding: utf-8 -*-
"""
KNN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import neighbors
from sklearn import metrics
from sklearn.preprocessing import minmax_scale

data=pd.read_excel(r'C:\Users\Administrator\Desktop\myGit\MachineLearning\KNN\CCPP.xlsx')
#数据归一化，消除量纲影响
predictors=data.columns[:-1]
X=minmax_scale(data[predictors])
#拆分为训练集和测试集
x_train,x_test,y_train,y_test=model_selection.train_test_split(X, data.PE,
                                                               test_size=0.25,random_state=1234)

设置k值集合
K=np.arange(1,int(np.ceil(np.log2(data.shape[0]))))
#存储不同k值的平均MSE
mse=[]
for k in K:
    #使用10折交叉验证的方法，比对每一k值下KNN模型的MSE
    cv_result=model_selection.cross_val_score(neighbors.KNeighborsRegressor(n_neighbors=k,
weights='distance'),x_train,y_train,cv=10,scoring='neg_mean_squared_error')
    mse.append((-1*cv_result).mean()) #将负数转换为正数
#查询最小均方误差的下标
arg_min=np.array(mse).argmin()

#绘制折线图
plt.rcParams['font.sans-serif']=['Microsoft YaHei'] #中文和负号正常显示
plt.rcParams['axes.unicode_minus']=False
plt.plot(K,mse) #折线图
plt.scatter(K,mse) #散点图
plt.text(K[arg_min],mse[arg_min],'最佳k值为%s' % int(K[arg_min]))
plt.show() #显示图形

#以最佳K值构建模型
knn_reg=neighbors.KNeighborsRegressor(n_neighbors=int(K[arg_min]),weights='distance')
knn_reg.fit(x_train,y_train)
predict=knn_reg.predict(x_test)

#模型评估
print('MSE:',metrics.mean_squared_error(y_test,predict)) #均方误差越小越好