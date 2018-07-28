# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:43:18 2018

@author: alexc
"""

from data_generators import ContinuousGenerator
from netmechanism import OutcomeSpaceGenerator, Sampler, SyntheticDataGenerator 
import time

# from multiprocessing import Pool, current_process 
# import multiprocessing.util as util
# util.log_to_stderr(util.SUBDEBUG)

def second_order_moment_experiment(dimensionality = 2, num_records = 20, test_frac = 0.5, batch_size = 7500, directory = '',\
                                   parallel = True, save_data = False, partition_method = 'fast_2', workers = -1, \
                                   sampling_workers = -1, num_samples = 25, samples_only = False, sample_parallel = True, load_data = False,\
                                   num_points_targets = 10, num_points_features = 10, epsilon = 0.1, seed = 23, allow_overwrite = False):
    # Initialise private_data object
    # '__spec__' = None
    private_data = ContinuousGenerator(d = dimensionality, n = num_records)
    private_data.generate_data(test_frac = test_frac, seed = seed)
    
    # Initialise OutcomeSpaceGenerator()
    OutcomeSpace = OutcomeSpaceGenerator(directory = directory, batch_size = batch_size, parallel = parallel,\
                                                  workers = workers, partition_method = partition_method, save_data = save_data)
    
    # Initialise Sampler() object
    SamplerInstance = Sampler(num_samples = num_samples, sampling_workers = sampling_workers, partition_method = partition_method, \
                              samples_only = samples_only, sample_parallel = sample_parallel, \
                              load_data = load_data)
    
    # Initialise SyntheticDataGenerator() object 
    SyntheticData = SyntheticDataGenerator(private_data, OutcomeSpace, SamplerInstance, privacy_constant = epsilon,\
                                           num_points_features = num_points_features, num_points_targets = num_points_targets, \
                                           seed = seed, allow_overwrite = allow_overwrite)
     
    t_start = time.time()
    results = SyntheticData.generate_data(property_preserved = 'second_moments')
    t_end = time.time()
    
    if parallel == True:
        print("Elapsed time with " + str(workers) + " workers is " + str(t_end - t_start))
    else:
        print("Elapsed time without parallelisation is " + str(t_end - t_start))
    
    # Synthetic data 
    # synthetic_data_integrated = SyntheticData.synthetic_datasets
    return results
    
    
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

if __name__ == '__main__':
     # Define experiment parameters
    dimensionality = 2
    num_records = 40
    test_frac = 0.5
    batch_size = 250
    # directory = '/homes/ac2123/Thesis'
    directory = 'D:/Thesis/Experiments'
    parallel = False
    save_data = False
    partition_method = 'fast_2'
    workers = -1
    sampling_workers = -1
    allow_overwrite = False
    num_samples = 50
    sample_parallel = False
    load_data = False
    num_points_targets = 5
    num_points_features = 5
    epsilon = 10
    seed = 23
    data = second_order_moment_experiment(dimensionality = dimensionality, num_records = num_records, batch_size = batch_size, \
                                            directory = directory, parallel = parallel, save_data = save_data, partition_method = partition_method, \
                                            workers = workers, sampling_workers = sampling_workers, num_samples= num_samples, sample_parallel = sample_parallel, load_data = load_data, \
                                            num_points_targets = num_points_targets, num_points_features = num_points_features, epsilon = epsilon, \
                                            seed = seed, allow_overwrite = allow_overwrite)




