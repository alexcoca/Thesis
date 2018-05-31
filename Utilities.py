# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:16:51 2018

@author: alexc
"""

import math
import numpy as np

def find_nearest1(array,value):
    '''Find the closest element to a value in an array. 
    Params: 
        @array: sorted array 
        @element: element whose closest index is sought'''
    idx = np.searchsorted(array,value,side="left")
    if idx > 0 and ( idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def find_nearest(array,value):
    idx = np.searchsorted(array,value,side="left")
    return idx

# Example testing code 
array = np.array([0.1,0.41,0.8,0.99,2])
element = 0.788
index = find_nearest(array,element)

# Find the index of the closest element to a value in a 1D numpy array
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx