# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:43:18 2018

@author: alexc
"""

from data_generators import ContinuousGenerator
from netmechanism import OutcomeSpaceGenerator, Sampler
from synthethic_data_generators import SyntheticDataGenerator
import time

# Initialise private_data object
dimensionality = 2
num_records = 20
private_data = ContinuousGenerator(d = dimensionality, n = num_records)
private_data.generate_data()

# Initialise OutcomeSpaceGenerator()
batch_size = 500
directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/'
parallel = False
workers = 2
partition_method = 'fast'
OutcomeSpaceGenerator = OutcomeSpaceGenerator(directory = directory, batch_size = batch_size, parallel = parallel,\
                                              partition_method = partition_method)

# Initialise Sampler() object
num_samples = 1
seed = 23
Sampler = Sampler(num_samples = num_samples, seed = seed, partition_method = partition_method)

# Initialise SyntheticDataGenerator() object 
num_points_targets = 5
num_points_features = 8
epsilon = 0.1
SyntheticDataGenerator = SyntheticDataGenerator(private_data, OutcomeSpaceGenerator, Sampler = [],\
                                                 privacy_constant = epsilon, num_points_features = num_points_features,
                                                 num_points_targets = num_points_targets)

# if '__name__' == 'main': 
t_start = time.time()
SyntheticDataGenerator.generate_data(property_preserved = 'second_moments')
t_end = time.time()

if parallel == True:
    print("Elapsed time with " + str(workers) + " workers is " + str(t_end - t_start))
else:
    print("Elapsed time without parallelisation is " + str(t_end - t_start))



