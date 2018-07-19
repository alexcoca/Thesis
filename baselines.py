# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:31:11 2018

@author: alexc
"""

import numpy as np
import mlutilities as utils

class DPRegression():
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
        self.rho = 0.01
        self.epsilon = 0.1
        self.X_bound = 1
        self.y_bound = 1
        
        
    def set_privacy_parameters(self, epsilon, num_records, X_bound = 1, y_bound = 1):
        ''' This method can be used to set privacy parameter epsilon. Delta is automatically set.
        @ Parameters:
            epsilon: privacy budget
            n: number of data points
            X_bound: feature space bound (sup ||x||_2 over the data universe X)
            y_bound: target space bound (abs(y) if y is scalar, otherwise as per X_bound ''' 
        self.epsilon = epsilon
        self.delta = np.minimum(1e-6, 1/num_records)
        self.rho = 0.01 # Bound on the error for ADASSP hold with prob 1 - rho
        self.X_bound = X_bound
        self.y_bound = y_bound
        
    def release_regression_coef(self, X, y):
        ''' This method releases regression coeffients with differential privacy.
        The method used is sufficient statistics pertrurbation with adaptive damping (#TODO: add ref)
        '''    
            
        # Helper constants
        priv_const = (np.sqrt(np.log(6/self.delta))/(self.epsilon/3))*(self.X_bound**2)
        d = int(np.shape(X)[1])
        
        # Calculate covariance matrix
        Sigma = np.transpose(X)@X
        
        # Smallest eigenvalues of empirical cov. matrix
        lambda_min = np.min(np.linalg.eig(Sigma)[0]) 
        
        # Privately release smallest eigenvalue
        lambda_private = np.max([lambda_min + priv_const*np.random.normal() - priv_const*np.sqrt(6/self.delta),0])
        
        # Calculate regularisation parameter
        reg_param = np.max([priv_const*np.sqrt(d*np.log(2*d**2/self.rho)) - lambda_private,0])
        
        # Privately release covariance matrix 
        Z = utils.generate_sym_rand_matrix(d)
        Sigma_private = Sigma + priv_const*Z
        
        # Privately release X^Ty 
        Z = np.random.normal(size=(d, 1))
        vec_priv = np.transpose(X)@y + (priv_const/self.X_bound)*self.y_bound*Z
        
        # Output regression parameters
        reg_parameters = np.linalg.solve(Sigma_private + reg_param*np.eye(d), vec_priv)
        
        return reg_parameters
    
    def get_parameters(self, X, y, num_samples, epsilon, seed = 23):
        
        np.random.seed(seed)
        
        parameters = []
        # Set privacy paramters 
        self.set_privacy_parameters(epsilon, X.shape[0])
        
        # Release differentially private regression coefficients 
        for _ in range(num_samples):
            parameters.append(self.release_regression_coef(X, y))
        
        return np.array(parameters)
    
class Regression():
    
    def __init__(self):
        self.parameters = []
        self.singular_indices = []
    
    def fit_data(self, data):
        
        if len(data.shape) > 2:
            
            parameters = np.zeros(shape = (*data.shape[:-1],1))
                        
            # Calculate empirical covariance matrix for all the data sets
            Sigma_tensor = np.transpose(data[:,:,:-1], axes = (0,2,1))@data[:,:,:-1] 
            # Determine if there are any singular synthetic matrices
            determinants = np.linalg.det(Sigma_tensor)
            mask = np.isclose(determinants, 0.0)
            if np.any(mask):
                print("Warning, there were singular sythethic matrices")
            singular_indices = np.nonzero(mask)
            mask = np.logical_not(np.isclose(determinants, 0.0))
            # Calculate features-targets correlations for all the data sets
            correlations = np.transpose(data[:,:,:-1], axes = (0,2,1))@data[:,:,-1:]
            parameters[mask] = np.linalg.solve(Sigma_tensor[mask,:,:], correlations[mask])
            
            # Least squares solution 
            for index in singular_indices[0]:
                parameters[index] = np.linalg.lstsq(Sigma_tensor[index,:,:], correlations[index])[0]
                
            self.singular_indices = singular_indices
            
        else:
            raise NotImplementedError
        
        return parameters 
        
    def calculate_predictive_error(self, test_data, model_parameters):
        
        if len(model_parameters.shape) > 2:
            
            # Calculate targets
            tmp = np.tile(test_data[:,:-1],[model_parameters.shape[0], 1, 1])*\
                                np.transpose(model_parameters, axes = (0, 2, 1))
            predicted_targets = np.sum(tmp, axis = 2, keepdims = True)
            rms_errors = np.sqrt((1/test_data.shape[0])*np.sum((test_data[:,-1:] - predicted_targets)**2, axis = 1))
            
        else:
            raise NotImplementedError
            
        return rms_errors
    
        
        