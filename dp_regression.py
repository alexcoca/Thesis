# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:31:11 2018

@author: alexc
"""

import numpy as np
import Utilities as utils

class Adassp():
    def __init__(self):
        '''Params 
        @epsilon: privacy budget
        @delta: probability that the mechanism output distribution is 
        significantly different for neighbouring databases
        @X_bound: defined as sup norm(x) where x is an element of the dataset
        @y_bound: defined as sup norm(y) (or abs(y) if y is scalar)'''
        self.epsilon = 0.1
        self.delta = 1e-6
        self.X_bound = 1
        self.Y_bound = 1
        self.X = []
        self.y = []
        self.reg_parameters = []
        np.random.seed(seed=5)
        
    def set_privacy_parameters(self,epsilon,n,X_bound=-1,y_bound=-1):
        ''' This method can be used to set privacy parameter epsilon. Delta is automatically set.
        @ Parameters:
            epsilon: privacy budget
            n: number of data points
            X_bound: feature space bound (sup ||x||_2 over the data universe X)
            y_bound: target space bound (abs(y) if y is scalar, otherwise as per X_bound ''' 
        self.epsilon = epsilon
        self.delta = np.min(1e-6,1/n[0])
        self.rho = 0.01 # Bound on the error for ADASSP hold with prob 1 - rho
        self.X_bound = X_bound
        
    def release_regression_coef(self,X,y):
        ''' This method releases regression coeffients with differential privacy.
        The method used is sufficient statistics pertrurbation with adaptive damping (#TODO: add ref)
        '''    
        
        # Compute feature space bound 
        if self.X_bound == -1:
            self.X_bound = utils.compute_bound(X)
        
        # Compute target space bound
        if self.y_bound == -1:
            self.y_bound = utils.compute_bound(y)
        
        # Calculate covariance matrix
        Sigma = np.transpose(X)@X
        
        # Helper constants
        priv_const = (np.sqrt(np.log(6/self.delta))/(self.epsilon/3))*self.X_bound**2
        d = int(np.shape(X)[1])
        
        # Smallest eigenvalues of empirical cov. matrix
        lambda_min = np.min(np.linalg.eig(Sigma)[0]) 
        
        # Privately release smallest eigenvalue
        lambda_private = np.max([lambda_min + priv_const*np.random.normal()  - priv_const*np.sqrt(6/self.delta),0])
        
        # Calculate regularisation parameter
        reg_param = np.max([priv_const*np.sqrt(d*np.log(2*d**2/self.rho))- lambda_private,0])
        
        # Privately release covariance matrix 
        Z = utils.generate_sym_rand_matrix(d)
        Sigma_private = Sigma + priv_const*Z
        
        # Privately release X^Ty 
        Z = np.random.normal(size=(d,))
        vec_priv = np.traspose(X)@y + (priv_const/self.X_bound)*self.y_bound*Z
        
        # Output regression parameters
        reg_parameters = np.linalg.solve(Sigma_private+reg_param*np.eye(d),vec_priv)
        
        return reg_parameters