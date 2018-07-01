# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:48:42 2018

@author: alexc
"""

from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))