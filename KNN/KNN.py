#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time
import threading
import random


class myT(object):
    def __init__(self, n):
        t1 = threading.Thread(target=self.th1, args=(n,))
        t2 = threading.Thread(target=self.th2, args=(n,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def th1(self, m):
        while 1:
            for i in range(m, -1, -1):
                print(1 / i)
                time.sleep(random.random() * 2)

    def th2(self, m):
        while 1:
            for i in range(m):
                print(-i)
                time.sleep(random.random() * 2)
