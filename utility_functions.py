# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 13:51:52 2018

@author: alexc
"""

import numpy as np

def compute_second_moment_utility(targets, outcomes, dim, scaling_const, F_tilde_x):
    """ Calculates the utility function for the sets of synthethic features stored in 
    the tensor outcomes, given the synthetic target vectors stored in the rows of the 
    array @targets. Other parameters are:
        @ dim, the dimensionality of the private data
        @ scaling_const: equal to the inverse global sensitivity of the utility function 
        times half the privacy constant, epsilon
    """
                        
    # Calculate f_r = 1/d Xh'y ((4.2), Chap. 4) where ' denotes transpose. Each y is a row of self.targets, 
    # so all X'y vectors are obtained as the rows of YX matrix. This is applied for all Xh 
    # in the synth_features_tensor
    f_r_tensor = (1/dim)*np.matmul(targets,outcomes)
    
    # Calculate F_r = 1/d Xh'Xh (' denotes transpose). This is applied for all Xh in the synth_features_tensor
    F_r_tensor = (1/dim)*np.transpose(outcomes, axes = (0,2,1))@outcomes
    
    # Perform tensor expansions so that F_tilde_r = [F_r_tensor,f_r_tensor] can be constructed.
            
    # f_r_tensor is expanded such that each component matrix is split into a tensor containing each
    # row transposed (a p x dim x 1 tensor, p is the number of targets)
    f_r_expand = f_r_tensor.reshape(tuple([*f_r_tensor.shape,1]))
    
    # F_r_tensor needs to be expanded such that each row of the corresponding entry in
    # f_r_tensor is appended as its last column and thus form all possible F_rs given a synethethic matrix Xh.
    # Hence each matrix in F_r is repeated p times to form a tensor (of p x d x d), where p is the number of rows of the 
    # corresponding YhXh matrix in f_r_tensor (which is = to the number of targets)
    F_r_expand = np.repeat(F_r_tensor,repeats=targets.shape[0],axis=0).reshape(F_r_tensor.shape[0],-1,*F_r_tensor[0].shape)
    
    # The concatenation results in a (b x p x d x d+1) tensor where p is the number of possible targets and b is the
    # batch size. This tensor is subtracted from f_tilde and reduced to obtain an a matrix of dimension batch_size x p 
    F_tilde_r = np.concatenate((F_r_expand,f_r_expand),axis=3)
    
    # Utilities for the particular batch are returned as a matrix of dimension batch_size x p where p is the number of 
    # synthetic targets. Exp-normalise trick is implemented so the exponentiation is done in the sampling step
    scaled_utilities = - scaling_const*np.max(np.abs(F_tilde_x-F_tilde_r), axis=(3,2))
    
    return scaled_utilities

def compute_first_moment_utility(targets, outcomes, dim, scaling_const, F_tilde_x):
    
    raise NotImplementedError

