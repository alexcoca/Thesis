# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:43:18 2018

@author: alexc
"""

from data_generators import ContinuousGenerator
from netmechanism import OutcomeSpaceGenerator, Sampler, SyntheticDataGenerator 
import time

# from multiprocessing import Pool, current_process 
import multiprocessing.util as util
util.log_to_stderr(util.SUBDEBUG)


if __name__ == '__main__':
    # Initialise private_data object
    # '__spec__' = None
    dimensionality = 3
    num_records = 20
    private_data = ContinuousGenerator(d = dimensionality, n = num_records)
    private_data.generate_data()
    
    # Initialise OutcomeSpaceGenerator()
    batch_size = 7500
    directory = 'D:/Thesis/Experiments'
    parallel = True
    workers = 6 # number of worker processes
    partition_method = 'slow'
    OutcomeSpaceGenerator = OutcomeSpaceGenerator(directory = directory, batch_size = batch_size, parallel = parallel,\
                                                  workers = workers, partition_method = partition_method)
    
    # Initialise Sampler() object
    num_samples = 5
    seed = 23
    samples_only = False
    SamplerInstance = Sampler(num_samples = num_samples, seed = seed, partition_method = partition_method, samples_only = samples_only)
    
    # Initialise SyntheticDataGenerator() object 
    num_points_targets = 10 # Number of points for interval discretisation
    num_points_features = 10 # Number of points for l2-lattice discretisation
    epsilon = 0.1
    SyntheticDataGenerator = SyntheticDataGenerator(private_data, OutcomeSpaceGenerator, Sampler = SamplerInstance,\
                                                     privacy_constant = epsilon, num_points_features = num_points_features,
                                                     num_points_targets = num_points_targets)
     
    t_start = time.time()
    SyntheticDataGenerator.generate_data(property_preserved = 'second_moments')
    t_end = time.time()
    
    if parallel == True:
        print("Elapsed time with " + str(workers) + " workers is " + str(t_end - t_start))
    else:
        print("Elapsed time without parallelisation is " + str(t_end - t_start))
    
    # Synthetic data 
    synthetic_data_integrated = SyntheticDataGenerator.synthetic_datasets

## Sample more data 
    
#params = SyntheticDataGenerator.sampling_parameters
## Re-initialise Sampler() object 
#num_samples  = 5
#partition_method = 'fast'
#seed = 24
#samples_only = True
#sampling_parameters = params
#AuxilliarySampler = Sampler(num_samples = num_samples, partition_method = partition_method, seed = seed, 
#                     samples_only = True, sampling_parameters = params)
#AuxilliarySampler.sample()
#new_samples = AuxilliarySampler.sampled_data_sets

    




