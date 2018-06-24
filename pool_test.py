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
# util.log_to_stderr(util.SUBDEBUG)

class DummySpaceGenerator():
    
    def __init__(self,features,dimensionality,folder_name):
        self.features = features
        self.res = []
        self.dimensionality = dimensionality
        self.folder_name = folder_name
        
    def simple_function(self,combination):
        partial_results = []
        ''' This dummy function returns an array containing the 
        sum of each row in the features for each row index in @index_array '''
        # print("Entering simple function")
        # print("Combinations batch",combination,current_process().name)
        res = np.sum(self.features[list(combination),:],axis=1)
        partial_results.append(res)
        # print("Result",res)
        fname = self.folder_name+current_process().name+".txt"
        # print ("Folder name",fname)
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%c')
        with open(fname,"a") as f:
            f.write(tstamp+" "+str(combination)+"\n")
        # print("Partial_results "+current_process().name,partial_results)

    
    def generate_dummy_space(self,chunksize,workers):
        # Create all combinations
        all_combinations = itertools.combinations(range(self.features.shape[0]),self.dimensionality)
        print ("Starting parallel pool:")
        pool = Pool(workers) # initialise a Process for each worker        
        results = pool.imap(self.simple_function,all_combinations,chunksize)
        pool.close()
        pool.join()
        # print ("Finish_time",datetime.datetime.fromtimestamp(time.time()).strftime('%c'))
        
           
# Create a toy 2-D array with n rows and d columns. Value along each row is const.
# while value along each column increases by 1
def test_parallel_pool(n,d,chunksize,workers):
    dummy_array = np.ones((n,d))
    multiplier =  np.expand_dims(np.arange(n),axis=1)
    dummy_array = multiplier*dummy_array
    folder_name = "C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/"
    #pdb.set_trace()
    t_start = time.time()
    if __name__=='__main__':
        #__spec__=None
        DummyGenerator = DummySpaceGenerator(dummy_array,d,folder_name)
        DummyGenerator.generate_dummy_space(chunksize,workers)
    t_end = time.time()
    print ("Time elaplsed with a chunksize of "+str(chunksize)+", "+str(workers)+" workers for n="+str(n)+" and d="+str(d)+" is "+str(t_end-t_start))

n = 30
d = 4
chunksize = 2500
#workers = 1
#test_parallel_pool(n,d,chunksize,workers)

chunksizes = [1,10,100,250,500,1000,5000,10000]
workers = [1,2,4,8]
for num_workers in workers[0:1]:
    for chunksize in chunksizes:
        test_parallel_pool(n,d,chunksize,num_workers)
    
    