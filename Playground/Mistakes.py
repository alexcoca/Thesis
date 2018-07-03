# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:55:15 2018

@author: alexc
"""
import numpy as np
# The function below does not actually work for d > 3. 
# The issue is how the indices are generated. np.tril/u generate the pairs
# [row,column] indices for the elements of the upper/lower triangular matrices
# To get a symmetric matrix you would need to generate the colum
def generate_sym_rand_matrix(d):
    '''This helper generates a symmetric random matrix of size d x d'''
    Z = np.zeros((d,d))
    idx_u = np.triu_indices(d,1)
    idx_l = np.tril_indices(d,-1)
    idx_l = tuple([idx_l[0][np.argsort(idx_l)[1]],np.sort(idx_l[1])])
    idx_diag = np.diag_indices(d)
    off_diag_entries = np.random.normal(size=(int((d*(d-1)/2)),))
    diag_entries = np.random.normal(size=(int(d),))
    Z[idx_u] = off_diag_entries
    Z[idx_l] = off_diag_entries
    Z[idx_diag] = diag_entries
    return Z