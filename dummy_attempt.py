# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:49:49 2018

@author: alexc
"""

import itertools
from multiprocessing import Pool,current_process 
from scipy.special import comb
import math
import multiprocessing.util as util
import time,datetime
import numpy as np

util.log_to_stderr(util.SUBDEBUG)

class Slicer():
    
    def __init__(self,n,k,batch_size,folder_name):
        self.n = n
        self.k = k
        self.combinations = itertools.combinations(range(n),k)
        self.batch_size = batch_size
        self.folder_name = folder_name
        # self.results = []
        self.max_util = 0
    
    def test_batches(self,batch,slice_integer):
        ''' Test that the slice_combinations gets batches of the correct size '''
        # Shows that we could use the slice integer to track the output
        fname = self.folder_name+current_process().name+"_"+str(slice_integer)+".txt"
        tstamp = datetime.datetime.fromtimestamp(time.time()).strftime('%c')
        with open (fname,'a') as f:
                for element in batch:
                    f.write(tstamp + " " + str(element) + "\n")
    
    def slice_combinations(self,slice_integer):
        # get batch of combinations
        batch = itertools.islice(self.combinations,(slice_integer)*self.batch_size,(slice_integer+1)*self.batch_size)
        # Testing, write batch sizes/or combinations to file to check the input to slice_combinations is correct
        self.test_batches(batch,slice_integer)
        if slice_integer <= 5:
            return 0.0
        elif slice_integer > 5 and slice_integer <= 10:
            return 1.0
        elif slice_integer > 10 and slice_integer < 50:
            return 2.0
        else:
            return 3.0

        
    
    def parallel_batch_processing(self,workers):
        
        # Calculate the number of batches
        n_batches = math.ceil(comb(self.n,self.k,exact=True)/self.batch_size)
        print("Number of batches is ",n_batches)
        print("Starting parallel pool")
        pool = Pool(workers)
        results = pool.imap(self.slice_combinations,range(n_batches))
        pool.close() # prevent further tasks from being submitted to the pool. Once all tasks have been
        # completed the worker processes will exit
        pool.join() # wait for the worker processes to exit. One must call close() before using join()
        print("Calculating max utility")
        self.max_util = max(results)


if __name__=='__main__':
    t_start = time.time()
    workers = 1
    test_object = Slicer(n=50,k=4,batch_size=2500,folder_name="C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/")
    test_object.parallel_batch_processing(workers)
    t_end = time.time()
    print("Elapsed time with "+str(workers)+" workers is "+str(t_end-t_start))
        


        
        