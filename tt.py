#!/usr/bin/env python
# -*- coding:utf-8 -*-
class commodity(object):
    def __init__(self, id, name, price):
        self.id = id
        self.name = name
        self.price = price

    def change_price(self, new_price):
        self.price = new_price

    def print_price(self):
        print(self.price)