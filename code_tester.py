#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:25:33 2018

@author: ac2123
"""

import second_moment_experiments_main as experiment

# Define experiment parameters
dimensionality = 3
num_records = 20
batch_size = 7500
directory = '/homes/ac2123/Thesis'
parallel = True
save_data = False
partition_method = 'fast_2'
workers = -1
num_samples = 25
sample_parallel = True
load_data = False
num_points_targets = 10
num_points_features = 10
epsilon = 0.1
seed = 23

experiment.second_order_moment_experiment(dimensionality = dimensionality, num_records = num_records, batch_size = batch_size, \
                                          directory = directory, parallel = parallel, save_data = save_data, partition_method = partition_method, \
                                          workers = workers, num_samples = num_samples, sample_parallel = sample_parallel, load_data = load_data, \
                                          num_points_targets = num_points_targets, num_points_features = num_points_features, epsilon = epsilon, \
                                          seed = seed)