
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:16:51 2018

@author: alexc
"""

import math
import numpy as np
import itertools

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

def get_unique_records(X,number=True,indices=False):
    ''' Returns a matrix containing the unique entries of matrix X. 
        If @number is True, then only the number of unique records is returned
        If @indices is True, then the indices of the unique rows of matrix X are 
        also returned. The output format is a tuple where the first entry represents
        the unique entries of matrix X (or their number depending on the value of the 
        @number value) and the second represents an array containing the indices of the 
        unique rows of matrix X.'''
    if number:
        if indices:
            unique = np.unique(X,return_index=True,axis=0)
            out = (unique[0].shape[0],unique[1])
        else:
            out = np.unique(X,axis=0).shape[0]
    else:
        if indices:
            unique = np.unique(X,return_index=True,axis=0)
            out = (unique[0],unique[1])
        else:
            out = np.unique(X,axis=0) 
    return out

def findsubsets(S,m):
    '''Returns all possible subsets of order m of a set S.
    The output format is a list of lists, where each list represents a subset of S.'''
    subsets = [list(x) for x in itertools.combinations(S,m)]
    return subsets

def bound_records_norm_deprecated(X):
    '''This function applies a transformation to a matrix X of n data records
    that ensures that each data record has a norm less than 1.
    
    @WARNING: This makes all features positive, which might be undesirable.'''
    max_features = np.max(X,axis = 0)
    min_features = np.min(X,axis = 0)
    denum = (max_features-min_features)*np.sqrt(X.shape[1])
    X = (X - min_features)/denum
    return X 

def bound_records_norm(X):
    '''This function applies a transformation to a matrix X of n data records
    that ensures that each data record has a norm less than 1.'''
    
    # Find the max norm of each row
    max_row_norm = np.max(np.linalg.norm(X, ord = 2, axis = 1))
    
    # Divide each row by the max row norm to obtain X
    X = X / max_row_norm    
    
    return X
