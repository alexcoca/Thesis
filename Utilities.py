
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:16:51 2018

@author: alexc
"""

import math
import numpy as np

def find_nearest_element(array,value):
    '''Find the closest element to a value in an array. 
    Params: 
        @array: sorted array 
        @element: element whose closest index is sought
    WARNING: NOT TESTED'''
    idx = np.searchsorted(array,value,side="left")
    if idx > 0 and ( idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def find_nearest_index(array,value):
    ''' Find the index of the closest element to a value in a 1D numpy array 
    WARNING: NOT TESTTED'''

    idx = np.searchsorted(array,value,side="left")
    return idx

# 
def find_nearest_idx(array,value):
    ''' Find the index of the closest element to a value in a 1D numpy array '''
    idx = (np.abs(array-value)).argmin()
    return idx

def generate_sym_rand_matrix(d):
    '''This helper generates a symmetric random matrix of size d x d'''
    Z = np.zeros((d,d))
    idx_u = np.triu_indices(d,1)
    idx_diag = np.diag_indices(d)
    off_diag_entries = np.random.normal(size=(int((d*(d-1)/2)),))
    diag_entries = np.random.normal(size=(int(d),))
    Z[idx_u] = off_diag_entries
    Z = Z + Z.T
    Z[idx_diag] = diag_entries
    return Z

def compute_bound(x):
    ''' This is technically the bound on the data universe X which I aproximate using the 
    training data. Assumes x is an nxd array where n is number of data points'''
    if np.shape(x) > 1:
        return np.max(np.linalg.norm(x,ord=2,axis=1))
    else: 
        return np.max(np.abs(x))

def get_unique_records(X):
    return np.unique(X,axis=0)