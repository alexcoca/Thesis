# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 18:51:33 2018

@author: alexc
"""

import numpy as np
import itertools
from multiprocessing import Pool,current_process
import multiprocessing.util as util
import time,datetime 
import os
util.log_to_stderr(util.SUBDEBUG)
from netmechanism import FeaturesLattice,TargetsLattice
from data_generators import ContinuousGenerator

class DummySpaceGenerator():
    
    def __init__(self,features,targets,private_data,dimensionality,epsilon,folder_name):
        self.features = features
        self.targets = targets
        self.res = []
        self.dimensionality = dimensionality
        self.folder_name = folder_name
        self.private_data = private_data
        self.F_tilde_x = 0
        self.synth_data_type = 'second_moments'
        self.epsilon = epsilon
        self.combinations = []
        self.batch_size = 10
        self.scaling_const = self.epsilon*self.private_data.features.shape[0]/4
        self.experiment_name = "eps"+str(self.epsilon).replace(".","")+"d"+\
                                str(self.dimensionality)#+"n1"+"n2"+"np"
        # TODO: Finish implementation of the experiment_name_property
        
    def compute_outcome_utility(self,outcomes):
        ''' This function computes the utility of the synthethic data sets
        stored in @outcome tensor as a function of the private data, stored in self.F_tilde_x
        (see thesis, pag. TBD). The returned value depends on which property the 
        synthethic data set preserves (self.synth_data_type). The value of the utility
        returned is normalised by its global sensitivity'''
        if self.synth_data_type == 'second_moments':
             
            # @outcome: a tensor containing a batch of synthethic feature matrices, Xh. 
    
            # Calculate f_r = 1/d Xh'y ((4.2), Chap. 4) where ' denotes transpose. Each y is a row of self.targets, 
            # so all X'y vectors are obtained as the rows of YX matrix. This is applied for all Xh 
            # in the synth_features_tensor
            
            f_r_tensor = (1/self.dimensionality)*np.matmul(self.targets,outcomes)
            
            # Calculate F_r = 1/d Xh'Xh (' denotes transpose). This is applied for all Xh in the synth_features_tensor
            F_r_tensor = (1/self.dimensionality)*np.transpose(outcomes,axes=(0,2,1))@outcomes
        
            # Perform tensor expansions so that F_tilde_r = [F_r_tensor,f_r_tensor] can be constructed.
            
            # 1. F_r_tensor needs to be expanded such that each row of the corresponding entry in
            # f_r_tensor is appended as its last column and thus form all possible F_rs given a synethethic matrix Xh.
            # Hence each matrix in F_r is repeated p times to form a tensor (of p x d x d), where p is the number of rows of the 
            # corresponding YhXh matrix in f_r_tensor
            
            # 2. f_r_tensor needs to be expanded such that each component matrix is split into a tensor containing each
            # row transposed (a pxdx1 tensor)
            
            # The concatenation results in a (b x p x d x d+1) tensor where p is the number of possible targets and b is the
            # batch size. This tensor is subtracted from f_tilde and reduced to obtain an a matrix of dimension batch_size x p 
            # where p is the number of potential targets
        
            utility = - np.max(np.abs(self.F_tilde_x - np.concatenate((np.repeat(F_r_tensor,repeats=self.targets.shape[0],axis=0).reshape(F_r_tensor.shape[0],-1,*F_r_tensor[0].shape), \
                                                                               f_r_tensor.reshape(tuple([*f_r_tensor.shape,1]))),axis=3)),axis=(3,2))
        else:
            pass # TODO: Placeholder for alternative utility function
            
        return utility
            
    def evaluate_sample_score(self,slice_index):
        ''' slice_index: To compute the large number of  
        combinations: itertools.slice object, obtained by taking
        slices in the interval [(@slice_index)*self.batch_size,(@slice_index+1)] 
        '''
        # Each tuple in combinations represents a choice of row indices from the
        # feature matrix. First, we obtain a slice from the entire combinations array
        batch = itertools.islice(self.combinations,(slice_integer)*self.batch_size,(slice_integer+1)*self.batch_size)
        
        # TODO: list(batch) is expensive. Could we replace self.combinations with a memory mapped array 
        # containing all the combinations and read from that instead? Would that be faster?
    
        # Create a tensor containing a batch of synthethic feature matrices, Xh. 
        
        self.compute_outcome_utility(self.features[list(batch),:])
        
        fname = self.folder_name+current_process().name+".txt"
        # print ("Folder name",fname)
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%c')
        # with open(fname,"a") as f:
        #    f.write(tstamp+" "+str(combination)+"\n")
        # print("Partial_results "+current_process().name,partial_results)

            #TODO: implement alternative once methodology complete
        return res
    
    def generate_dummy_space(self,synth_data_type,chunksize,workers):
        
        # Calculate utility contribution from the private data
        self.synth_data_type = synth_data_type
        
        if synth_data_type == 'second_moments':
            
            # Implement equation (4.3, Chap. 4) for private data. Only calculated once
            const = (1/self.private_data.features.shape[0])
            F_x = const*self.private_data.features.T@self.private_data.features
            f_x = const*self.private_data.features.T@self.private_data.targets
            self.F_tilde_x = np.concatenate((F_x,f_x),axis=1)
            
            # Inverse of global sensitivity
            igs = self.private_data.features.shape[0]/2
            self.scaling_const = igs*(self.epsilon/2)
        else:
            pass 
            # TODO: placeholder for other utilitities functions
            
        # Create all combinations
        self.combinations = itertools.combinations(range(self.features.shape[0]),self.dimensionality)
        
        # Start parallel pool
        print ("Starting parallel pool:")
        pool = Pool(workers) # initialise a Process for each worker        
        results = pool.imap(self.simple_function,all_combinations,chunksize)
        pool.close()
        pool.join()
        self.res = itertools.islice(results,100)
        # print ("Finish_time",datetime.datetime.fromtimestamp(time.time()).strftime('%c'))
        
           
# Create a toy 2-D array with n rows and d columns. Value along each row is const.
# while value along each column increases by 1
#def test_parallel_pool(n,d,chunksize,workers):
#    dummy_array = np.ones((n,d))
#    multiplier =  np.expand_dims(np.arange(n),axis=1)
#    dummy_array = multiplier*dummy_array
#    folder_name = "C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/"
#    #pdb.set_trace()
#    t_start = time.time()
#    if __name__=='__main__':
#        #__spec__=None
#        DummyGenerator = DummySpaceGenerator(dummy_array,d,folder_name)
#        DummyGenerator.generate_dummy_space(chunksize,workers)
#        results = DummyGenerator.res
#    t_end = time.time()
#    print ("Time elaplsed with a chunksize of "+str(chunksize)+", "+str(workers)+" workers for n="+str(n)+" and d="+str(d)+" is "+str(t_end-t_start))
#    return results

d = 2
chunksize = 1000
workers = 8
num_points_feat = 8
num_points_targ = 10
n = 20
#results = test_parallel_pool(n,d,chunksize,workers)

# Outcome space components
flattice = FeaturesLattice()
flattice.generate_l2_lattice(dim=d,num_points=num_points_feat)
tlattice = TargetsLattice()
tlattice.generate_lattice(dim=d,num_points=num_points_targ)
features = flattice.points
targets = tlattice.points

# Generate private data
private_data_generator = ContinuousGenerator(d,n)
private_data = private_data_generator.generate_data()



folder_name = "C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/"
#pdb.set_trace()
t_start = time.time()
if __name__=='__main__':
    #__spec__=None
    DummyGenerator = DummySpaceGenerator(features,targets,private_data,d,folder_name)
    DummyGenerator.generate_dummy_space(synth_data_type,chunksize,workers)
    results = DummyGenerator.res
t_end = time.time()    
print ("Time elaplsed with a chunksize of "+str(chunksize)+", "+str(workers)+" workers for n="+str(n)+" and d="+str(d)+" is "+str(t_end-t_start))

# Array X
n = 9
d = 3
X = np.arange(n*d).reshape(n,d)
# Create tensor y
combs = [(0,1,2),(0,1,3),(0,1,4),(0,1,5)]
y = X[combs,:]
# Add a dummy column of 1.0s to each element of the y tensor
b = np.array([1.0,1.0,1.0]).reshape(1,3)
b = b.repeat(y.shape[0],axis=0).reshape(y.shape[0],y.shape[1],1)
# Concatenate the column with the tensor
y_new = np.concatenate((y,b),axis=2)

matrix = np.ones((n,d))
multiplier =  np.expand_dims(np.arange(n),axis=1)
matrix = multiplier*matrix

def append_column(y,b):
    #  Shape b appropriately
    b = b.repeat(y.shape[0],axis=0).reshape(y.shape[0],y.shape[1],-1)
    # Concatenate the column with the tensor
    y_new = np.concatenate((y,b),axis=2)
    return y_new

def tensor_expansion(X,combs,matrix):
    y = X[combs,:]
    # Some recursive implementation?
    index = matrix.shape[0]
    if index == 0:
        # Concatenate last row of matrix to y
        res = append_column(y,b)
    else:
        #tensor_expansion(X,combs,....)

n = 9
d = 3

X = np.arange(n*d).reshape(n,d)
combs = [(0,1,2),(0,1,3),(0,1,4),(0,1,5)]
y = X[combs,:]
matrix = np.ones((n,d))
multiplier =  np.expand_dims(np.arange(n),axis=1)
matrix = multiplier*matrix

    
y = np.stack([y]*matrix.shape[0])
matrix_reshape = matrix.reshape(matrix.shape[0],matrix.shape[1],-1)
matrix_stack = np.stack([matrix_reshape]*X[combs,:].shape[0],axis=1)
y_new = np.concatenate((y,matrix_stack),axis=3)

f_expansion = y.reshape(tuple([*y.shape,1]))
T_expansion = np.repeat(y,repeats=y[0].shape[0],axis=0).reshape(y.shape[0],-1,*y[0].shape)
test = np.concatenate((T_expansion,f_expansion),axis=3)
# f_expansion = np.expand_dims(y)