#!/usr/bin/env python
# -*- coding:utf-8 -*-
from DataClean import data_clean
import numpy as np

x = [1,3,4,5,9]
y = np.array(x).reshape(-1, 1)
dc = data_clean()
cd = dc.normalization(y, 'z-score')
print(cd)