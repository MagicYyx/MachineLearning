#!/usr/bin/env python
# -*- coding:utf-8 -*-
# import numpy as np
import pandas as pd
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def cov(x,mu):
    s = 0
    m = x - mu
    for i in range(m.shape[1]):
        s += np.dot(m[:,i].reshape(x.shape[0],1),m[:,i].reshape(1,x.shape[0]))
    return s

path = os.path.join(os.getcwd(), 't.xlsx')
data = pd.read_excel(path)
X1 = np.array(data[data.y == 0].drop('y', axis=1)).T
X2 = np.array(data[data.y == 1].drop('y', axis=1)).T
mu1 = np.mean(X1, axis=1).reshape(X1.shape[0], 1)
mu2 = np.mean(X2, axis=1).reshape(X2.shape[0], 1)
cov1 = cov(X1,mu1)
cov2 = cov(X2,mu2)
Sw = np.mat(cov1 + cov2)
print(Sw.I*np.mat(mu1-mu2))
#
lda = LinearDiscriminantAnalysis()
lda.fit(data.drop('y', axis=1), data.y)
print(lda.intercept_)
print(lda.coef_)
