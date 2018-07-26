
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:24:09 2018

@author: alexc
"""

#Useful plotting indicatiors r--,bs,g^,b.,b*

#from Generators import ContinuousGenerator
#from loaders import DataLoader
#import matplotlib.pyplot as plt
#import numpy as np
#import math
#from netmechanism import FeaturesLattice
#import testutilities
#%% Test regression line using DataGenerator object
#data = generators.DataGenerator(reg_slope=1,reg_intercept=0,num_pts_y_lattice=10)
#line = data.generate_reg_line()
#plt.plot(line[:,0],line[:,1],'g^')
#%% Testing lattice generation
#data = generators.DataGenerator(reg_slope=2,reg_intercept=1,num_pts_y_lattice=10)
#lattice = data.generate_lattice()
#points = data.plot_lattice(lattice)
#%% Test that generate_data_set correctly sets the properties lattice and regression_line and x_coordinate generation
#generator = generators.DataGenerator()
#test = generator.lattice
#print(test)
#DataSet = generator.generate_data_set(num_pts=15,reg_slope=2,reg_intercept=1,num_pts_x_lattice=100,num_pts_y_lattice=50)
#lattice = generator.lattice
#regression_line = generator.regression_line
#x_coordinates = generator.x_locs
#print(lattice)
#print(regression_line)
#%% Test generation on that when more outputs are required compared to the number of lattice x coordinates
#generator = DataGenerator(batch_size=15)
#generator.generate_data_set(num_pts=60,reg_slope=2,reg_intercept=1,num_pts_x_lattice = 20,num_pts_y_lattice = 50)
#data_set = generator.data
#plt.figure(0)
#plt.scatter(data_set[:,0],data_set[:,1])
#generator = DataGenerator()
#generator.generate_data_set(num_pts=35,reg_slope=2,reg_intercept=1,num_pts_x_lattice = 100,num_pts_y_lattice = 50)
#data_set = generator.data
#plt.figure(1)
#plt.scatter(data_set[:,0],data_set[:,1])
#generator = DataGenerator()
#generator.generate_data_set(num_pts=15,reg_slope=3,reg_intercept=0.5,num_pts_x_lattice = 50,num_pts_y_lattice = 50)
#data_set = generator.data
#plt.figure(2)
#plt.scatter(data_set[:,0],data_set[:,1])
#%% Testing data loader
#path = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Data/raw/solar/flare.data.2.txt'
## Create data loader object and load data
##loader = DataLoader()
#features = list(range(3,9))
#targets = [10]
#loader = DataLoader()
#loader.load(path,feature_indices=features,target_indices=targets,unique=True,boundrec=True)
#data = loader.data
#print (data)
#loader = DataLoader()
#loader.load(path,feature_indices=features,target_indices=targets,unique=True)
#new_data = loader.data
#print(new_data)
#%% Test ContinuousGenerator class
## Create a generator object
#generator = ContinuousGenerator(d=1,n=10,seed=2)
## Generate data
#generator.generate_data(bound_recs=False)
#dataset = generator.data
#print(dataset)
#generator.plot_data()
#norms = np.linalg.norm(dataset,ord=2,axis=1)
#print(norms)
#%% Testing testuitlities filter functions
#dim = 2
#num_points = 25
#upper_bound = 1.0
#lower_bound = -1.0
#radius = 1.0
#intersection_m2,coord_array_m2 = testutilities.bruteNonIntegerIntersection(dim=dim,radius=radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound)
#tmp = testutilities.filter_unsorted(intersection_m2)
#filtered_results = testutilities.filter_signed(tmp)

#%% Test FeaturesLattice Class

from netmechanism import FeaturesLattice
import testutilities
import numpy as np
import math

dim = 2
num_points = 50
upper_bound = 1.0
lower_bound = -1.0
num_dec = 4
radius = 1.0
r_tol = 1e-5
OutputLattice = FeaturesLattice()
OutputLattice.generate_l2_lattice(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points,pos_ord=True,rel_tol=r_tol)
#intersection_m2 = testutilities.bruteNonIntegerIntersection(dim=dim,radius=radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound,filtered = False,r_tol=r_tol)
test_points = OutputLattice.points
# Points that are returned by the fancy algorithm but not by brute
#differences_1 = testutilities.get_differences(test_points,intersection_m2)
#assert differences_1.size == 0
## Points that are returned by the brute but not the fancy algorithm
#differences_2 = testutilities.get_differences(intersection_m2,test_points)
#assert differences_2.size == 0
## Test that all the solutions have the correct length
#lengths = [len(x) == dim for x in test_points]
#assert np.all(lengths)
## Test that all the solutions are unique
#assert np.unique(test_points,axis=0).shape[0] == test_points.shape[0]
## Test that the norms of the elements returned are correct
#norms = np.linalg.norm(np.array(test_points),ord=2,axis=1)
#close_norms = [True if math.isclose(np.linalg.norm(x),1,rel_tol=1e-7) == True else False for x in norms]
#small_norms = list(np.round(norms,decimals=num_dec) <=radius)
#all_norms = [x or y for x,y in zip(small_norms,close_norms)]
## incorrect_points = np.array(test_points)[np.logical_not(all_norms)]
#incorrect_points = [point for (indicator,point) in zip(np.logical_not(all_norms),test_points) if indicator==True]
#assert np.all(all_norms)
## Test that the two methods return the same number of solutions
#assert intersection_m2.shape[0] == len(test_points)
#%%  Testing TargetsLattice class
#from netmechanism import TargetsLattice
#from scipy.special import comb,factorial 
#
#num_points = 5
#dim = 3 
#
#TargetsLattice = TargetsLattice()
#TargetsLattice.generate_lattice(dim=dim,num_points=num_points)
#target_vectors = TargetsLattice.points
## Make sure you don't have duplicate solutions
#assert np.unique(target_vectors,axis=0).shape[0] == target_vectors.shape[0]
## Make sure the number of elements returned is correct
#num_elements = comb(num_points,dim)*factorial(dim)
#assert num_elements == target_vectors.shape[0]
#%% Testing save lattice functionality for FeaturesLattice class
#from netmechanism import FeaturesLattice
#import pickle
#
#dim = 3
#num_points = 12
#upper_bound = 1.0
#lower_bound = -1.0
#radius = 1.0
#num_dec = 10
#folder_path = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Lattices'
#lattice_name='d_3_npt_12'
#r_tol = 1e-5
#OutputLattice = FeaturesLattice()
#OutputLattice.generate_l2_lattice(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points,pos_ord=True,rel_tol=r_tol)
#original_data = OutputLattice.points
#OutputLattice.save_lattice(folder_path=folder_path,lattice_name=lattice_name)
#basepath = folder_path+"/"+lattice_name
#with open(basepath,"rb") as data:
#    reloaded_data = pickle.load(data)
#%%
# Test to see if slices can be pickled

#import itertools, pickle, os
#
#
#
##@profile
#def load_batch_scores(path):
#    ''' Returns the contents of the file specified by absolute path '''
#    with open(path,"rb") as data:
#        batch_scores = pickle.load(data)
#    return batch_scores
#
#def save_batch_scores(batch,directory,filename):
#    ''' Saves the batch of scores to the location specified by @directory with
#    the name specified by @lattice_name.'''
#    
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#        
#    full_path = directory+"/"+filename
#   
#    # Raise an error if the target file exists
#    if os.path.exists(full_path):
#        assert False
#       
#    with open(full_path,"wb") as data:
#        pickle.dump(batch,data)
#
#test = itertools.combinations(range(100),5)
#big_slice = itertools.islice(test,50000)
#
#experiment_name = 'dummy_experiment'
#filename = 'pickled_slice'
#directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/'+experiment_name+'/OutcomeSpace'
#path = directory+"/"+filename
#
#save_batch_scores(big_slice,directory,filename)
#print ("Slice size before pickling",len(list(big_slice)))
#unpickled_slice = load_batch_scores(path)
#print ("Slice size after pickling",len(list(unpickled_slice)))
#
#
## Now let's see what difference pickling the "unwrapped" iterator might have
#test = itertools.combinations(range(100),5)
#big_list = list(itertools.islice(test,50000))
#
#experiment_name = 'dummy_experiment'
#filename = 'pickle_slice_as_list'
#directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/'+experiment_name+'/OutcomeSpace'
#path = directory+"/"+filename
#
#save_batch_scores(big_list,directory,filename)
#print ("Len size before pickling",len(list(big_list)))
#unpickled_slice = load_batch_scores(path)
#print ("Len size after pickling",len(list(unpickled_slice)))

# Conclusion: Makes a massive difference, the list storage is ~850 kB for this
# case while the slice only 

# Can we pickle a tee? (Yes, we can)
#experiment_name = 'pickle_tee'
#filename = 'pickle_tee'
#directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/'+experiment_name+'/OutcomeSpace'
#path = directory+"/"+filename
#
#test = itertools.combinations(range(100),5)
#slice_o,slice_c = itertools.tee( itertools.islice(test,50000) )
#
#save_batch_scores(slice_c,directory,filename)
#print ("Tee elice size before pickling",len(list(slice_c)))
#print ("Check we have not affected the other slice",len(list(slice_o)))
#unpickled_slice = load_batch_scores(path)
#print ("Tee slice size after pickling",len(list(unpickled_slice)))



#%% 

# Tested to see which data structure supports faster iteration.
# Tuple seems faster if obj is created as a simple list. However, if obj is
# an iterator then it has to be recreated at every step to allow timing and then
# cast to a list. In this case list is only slightly faster --> choose tuple
#
#import numpy as np 
#from operator import itemgetter
## Create base tuple/list
#tup = (np.random.normal(),np.random.normal())
#lst = [np.random.normal(),np.random.normal()]
#
#def create_obj (obj,num_reps):
#    obj = iter[obj for i in range(num_reps)]
#    return obj
#    
#def perform_operation():
#    # obj = create_obj(lst,100000)
#    obj = list(obj)
#    partition_function = sum([elem[0] for elem in obj])
#    global_max = max(obj,key=itemgetter(1))[1]
#    return (partition_function,global_max)
#    
#test_obj_1 = create_obj(tup,100000)
#test_obj_2 = create_obj(lst,100000)
#%% Testing feature matrix recover algorithm
#import numpy as np
#import itertools
#import math 
#from scipy.special import comb 
#
#
#def grouper(iterable, n, fillvalue=None):
#    "Collect data into fixed-length chunks or blocks"
#    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
#    args = [iter(iterable)] * n
#    return itertools.zip_longest(*args, fillvalue=fillvalue)
#
#def construct_batches(n,k,batch_size):
#    
#    combinations_slices = []
#    
#    # Calculate number of batches
#    n_batches = math.ceil(comb(n,k,exact=True)/batch_size)
#    
#    # Construct iterator for combinations
#    combinations = itertools.combinations(range(n),k)
#    
#    # Slice the iterator into n_batches slices. Each slice is duplicated
#    # so that it can be subsequently written to a file for later retrieval 
#    # during sampling. This is necessary since calling the utility calculation
#    # routine exahausts the original slice. If the iterator is first converted to list
#    # then storage requirement increases (e.g. from 1KB to 800+ kB for a list of
#    # 50,000 tuples of dimension 4)
#    
#    while len(combinations_slices) < n_batches:
#        combinations_slices.append(itertools.islice(combinations,batch_size))
#        
#    return combinations_slices
#
#def nth(iterable, n, default=None):
#        "Returns the nth item from iterable or a default value"
#        return next(itertools.islice(iterable, n, None), default)
#
#n  = 10
#k = 4
#batch_size = 10
#combinations_slices = construct_batches(n,k,batch_size)
#combinations_numbers = [5,7,0]
#batches = [2,3,1]
#
## Define toy features
#dummy_array = np.ones((n,k))
#multiplier =  np.expand_dims(np.arange(n), axis = 1)
#features = multiplier*dummy_array
#
## Select only relevant slices
#combinations_slices = [combinations_slices[i] for i in range(len(combinations_slices)) if i in batches]
#
#for element in combinations_slices:
#    print(list(element))
#
## Feature matrices
#feature_matrices = [features[nth(combination_slice,combination_number),:] for combination_slice,combination_number in zip(combinations_slices,combinations_numbers)] 
#
## Check if the selected matrices correspond to the combinations
#combinations_slices = construct_batches(n,k,batch_size)
#
#selected_combinations = [nth(combination_slice,combination_number) for combination_slice,combination_number in zip(combinations_slices,combinations_numbers)]
#
## Check if the slices were correct
#combinations_slices = construct_batches(n,k,batch_size)

# Recovery incorrect - revealed the idea with creating the slices was wrong
#%%  Testing multiple choices for data structure saving on disk
#import numpy as np
#import os
#import pickle
#
#def save_batch_scores(batch, filename, directory = '', overwrite = False):
#    ''' Saves the batch of scores to the location specified by @directory with
#    the name specified by @filename.'''
#    
#    # If directory is not specified, then the full path is provided in filename
#    # This is used to overwrite the files containing the raw scores during the 
#    # calculation of the partition function to avoid unnecessary duplication of 
#    # operations during the sampling step
#    
#    if not directory:
#        full_path = filename
#        
#        if overwrite:
#            with open(full_path,"wb") as data:
#                pickle.dump(batch,data)
#        else:
#            # Overwriting data only alowed if this is explicitly mentioned
#            if os.path.exists(full_path):
#                assert False
#            
#    else:
#        
#        # Create directories if they don't exist
#        if not os.path.exists(directory):
#            os.makedirs(directory)
#            
#        full_path = directory+"/"+filename
#       
#        # Raise an error if the target file exists
#        if os.path.exists(full_path):
#                assert False
#           
#        with open(full_path,"wb") as data:
#            pickle.dump(batch,data)
#            
#def load_batch_scores(path):
#    ''' Returns the contents of the file specified by absolute path '''
#    with open(path,"rb") as data:
#        batch_scores = pickle.load(data)
#    return batch_scores
#
## Declare dummy data
#batch_size = 3000
#targets = 45
#dummy_scores = np.random.random(size=(batch_size,targets))
#dummy_batch_index = 3
#
#data_structures =[ {'scores':dummy_scores,'index':dummy_batch_index}, [dummy_scores,dummy_batch_index] ] 
#
#experiment_name = 'memory_footprint_test'
#directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/' + experiment_name + '/OutcomeSpace'
#filenames = ['dictionary','list']
#
#for data_structure, filename in zip(data_structures,filenames):
#    save_batch_scores(data_structure,filename,directory)
#
#dictionary_recovered = load_batch_scores(directory+"/dictionary")
#list_recoveredv = load_batch_scores(directory+"/list")

# Conclusion: Slight advantage to using a list, but disk space is likely not going to be an issue...

#%% Test to prove the recovery method works
#import itertools
#
#def nth(iterable, n, default=None):
#        "Returns the nth item from iterable or a default value"
#        return next(itertools.islice(iterable, n, None), default)
#
#
#n = 10
#k = 4
#batch_size = 10
#    
#batches = [2,3,1]
#combs_idxs = [5,3,0]
#
#features = []
#
#for batch_idx,comb_idx in zip(batches,combs_idxs):
#    recovered_slice = itertools.islice(itertools.combinations(range(n),k),(batch_idx)*batch_size,(batch_idx+1)*batch_size)
#    features.append(nth(recovered_slice,comb_idx))
#    
#combinations_list = list(itertools.combinations(range(n),k))


#%%  Testing tensor manipulations
from netmechanism import FeaturesLattice, TargetsLattice
import itertools, functools
import numpy as np
import math
from scipy.special import comb
from data_generators import ContinuousGenerator
import time
import pickle,os, glob
import testutilities
import collections
from baselines import Regression, DPRegression
          
def get_synthetic_F_tilde (synthetic_data):
    
    # Compute F_tilde (equation (4.1), Chapter 4, Section 4.1.1)
    # for the private data
    
    const = (1/dim)
    F_x = const*synthetic_data[:,:-1].T@synthetic_data[:,:-1]
    f_x = const*synthetic_data[:,:-1].T@synthetic_data[:,-1:]
    F_tilde_x = np.concatenate((F_x,f_x), axis = 1)
    
    return F_tilde_x  

#@profile
def calculate_partition_function(filenames,n_batches,test = False, max_score = 0.0):
    
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
    
    filenames = sorted(filenames, key = get_batch_id)
    
#    max_score = - math.inf
#
#    for result in iter(results):
#        if result[0] > max_score:
#            max_score = result[0]
    
#    print ("Max score evaluated inside calculate_partition_function", max_score)
    
    partition_function  = 0
    
    for batch in range(n_batches):
        data = testutilities.retrieve_scores(filenames,[batch])[0]
        
        if test:
            # Keep a copy of the original data to allow testing 
            fname = "orig_" + filenames[batch][filenames[batch].rfind("\\") + 1:]
            testutilities.save_batch_scores(data,fname,directory.replace("*",""))
            
        # Apply sum-exp trick when calc. partition to avod numerical issues
        data['scores'] = np.exp(data['scores'] - max_score)
        partition_function += np.sum(data['scores'])
        
        # Overwrite the raw batch scores with the scores calculated as per 
        # Step 3 of the procedudre in Chapter 4, Section 4.1.3 
        testutilities.save_batch_scores(data, filenames[batch], overwrite = True)
    
    return partition_function

# @profile
def calculate_partition_function_alt(iterable):
    
    def func(scores_batch,max_score):
        return np.sum(np.exp(scores_batch[1]-max_score))
        
    iter_1,iter_2 = itertools.tee(iterable)
    
    ms = functools.reduce((lambda x,y:(max(x[0],y[0]),np.zeros(shape=y[1].shape))),iter_1)[0]    
    
    print ("Max scaled utility evaluated by the alternative partition function calculation", ms)
    
    partition_function = sum(map(functools.partial(func,max_score=ms),iter_2))
    
    return partition_function

def sample_dataset_deprecated(n_batches, num_samples, partition_function, filenames, seed):
    
    np.random.seed(seed)
    
    def get_sample(scaled_partition):
        
        def get_sample_idxs(scores,scaled_partition):
            
            row_idx = 0      
            col_idx = 0 
            max_col_id = scores.shape[1] - 1
            cum_scores = np.sum(scores,axis=1)
            candidate_partition = scaled_partition
            print ("Calculating row and column, starting with partition", scaled_partition)
            while candidate_partition > 0:
                candidate_partition = scaled_partition - cum_scores[row_idx]
                if candidate_partition > 0:
                    scaled_partition = candidate_partition 
                    row_idx += 1
            print ("After rows contributions", scaled_partition)
            for element in scores[row_idx,:]:
                scaled_partition -= element
                if scaled_partition > 0:
                    # print ("The value of the score is", element)
                    col_idx += 1
                else:
                    if col_idx == 0:
                        col_idx = max_col_id
                        row_idx = row_idx - 1
                    else:
                        col_idx = col_idx - 1
                    print ("Final partition value is", scaled_partition)
                    break
            return (row_idx,col_idx)
        
        # Retrive the data for every batch and subtract from partition function
        
        orig_partition = scaled_partition
        print ("Starting from the scaled partition", scaled_partition)
        for batch in range(n_batches):
            scores = testutilities.retrieve_scores(filenames,batches=[batch])[0]['scores']
            candidate = scaled_partition - np.sum(scores)
            if candidate > 0:
                scaled_partition = candidate
            else: 
                print ("After batch contribution partition residual is", scaled_partition)
                row_idx, col_idx = get_sample_idxs(scores,scaled_partition)
                break
        return (batch, row_idx, col_idx, orig_partition)
   
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
        
    # Store sample indices
    sample_indices = []
     
    # Filenames have to be sorted to ensure correct batch is extracted
    filenames  = sorted(filenames, key = get_batch_id)
    
    for i in range(num_samples):
        
        # To sample, a random number in [0,1] is first generated and multiplied with the partition f
        # (Step 5, in Chapter 4, Section 4.1.3)
        scaled_partition = partition_function*np.random.random()
        # To get a sample, the scores from each batch are subtracted from the scaled partition
        # until the latter becomes negative. The data set for which this zero crossing is 
        # attained is the sampled value ( Step 6, Chapter 4, Section 4.1.3)
        sample_indices.append(get_sample(scaled_partition))
    print("Sampled indices, old algorithm with seed " + str(seed) + " are",sample_indices)
    return sample_indices

def sample_dataset(n_batches, num_samples, partition_function, filenames, seed):
    
    np.random.seed(seed)
    
    def get_sample(scaled_partition):
        
        def get_sample_idxs(scores,scaled_partition):
            
            row_idx = 0      
            col_idx = 0 
            max_col_id = scores.shape[1] - 1
            cum_scores = np.sum(scores,axis=1)
            candidate_partition = scaled_partition
            # print ("Calculating row and column, starting with partition", scaled_partition)
            while candidate_partition > 0:
                candidate_partition = scaled_partition - cum_scores[row_idx]
                if candidate_partition > 0:
                    scaled_partition = candidate_partition 
                    row_idx += 1
            # print ("After rows contributions", scaled_partition)
            for element in scores[row_idx, :]:
                scaled_partition -= element
                if scaled_partition > 0:
                    # print ("The value of the score is", element)
                    col_idx += 1
                else:
                    if col_idx > max_col_id:
                        row_idx = row_idx + 1
                        col_idx = 0 
                    print ("Final partition value is", scaled_partition)
                    break
            return (row_idx,col_idx)
        
        # Retrive the data for every batch and subtract from partition function
        
        orig_partition = scaled_partition
       # print ("Starting from the scaled partition", scaled_partition)
        for batch in range(n_batches):
            scores = testutilities.retrieve_scores(filenames,batches=[batch])[0]['scores']
            candidate = scaled_partition - np.sum(scores)
            if candidate > 0:
                scaled_partition = candidate
            else: 
               # print ("After batch contribution partition residual is", scaled_partition)
                row_idx, col_idx = get_sample_idxs(scores,scaled_partition)
                break
        return (batch, row_idx, col_idx, orig_partition)
   
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
        
    # Store sample indices
    sample_indices = []
     
    # Filenames have to be sorted to ensure correct batch is extracted
    filenames  = sorted(filenames, key = get_batch_id)
    
    for i in range(num_samples):
        
        # To sample, a random number in [0,1] is first generated and multiplied with the partition f
        # (Step 5, in Chapter 4, Section 4.1.3)
        scaled_partition = partition_function*np.random.random()
        # To get a sample, the scores from each batch are subtracted from the scaled partition
        # until the latter becomes negative. The data set for which this zero crossing is 
        # attained is the sampled value ( Step 6, Chapter 4, Section 4.1.3)
        sample_indices.append(get_sample(scaled_partition))
    print("Sampled indices, old algorithm with seed " + str(seed) + " are",sample_indices)
    return sample_indices

def calculate_accuracy(synthetic_datasets, dimensionality, F_tilde_x):
        
    # Calculate empirical covariance and correlation for the synthetic data sets.
    # Each data set is a member of the tensor @self.synthetic_datasets
    emp_cov_synth = 1/dimensionality * np.transpose(synthetic_datasets[:,:,:-1], axes = (0,2,1))@synthetic_datasets[:,:,:-1]
    emp_corr = 1/dimensionality * np.transpose(synthetic_datasets[:,:,:-1], axes = (0,2,1))@synthetic_datasets[:,:,-1:]
    
    # Calculate differences between private data empirical covariance and feature/targets correlation and
    # their synthetic counterparts
    delta_cov = F_tilde_x[:,:-1] - emp_cov_synth
    delta_corr = F_tilde_x[:,-1:] - emp_corr

    # Calculate norms of the differences 
    delta_cov_norms_f = np.linalg.norm(delta_cov, ord = 'fro', axis = (1,2))
    delta_cov_norms_2 = np.linalg.norm(delta_cov, ord = 2, axis = (1,2))
    delta_corr_norms_2 = np.linalg.norm(delta_corr, ord = 2, axis = 1)

    # Calculate average and standard deviation of the norm of the differences
    avg_f_norm_cov = np.mean(delta_cov_norms_f)
    
    avg_2_norm_cov = np.mean(delta_cov_norms_2)
    std_f_norm_cov = np.std(delta_cov_norms_f)
    std_2_norm_cov = np.std(delta_cov_norms_2)
    avg_2_norm_corr = np.mean(delta_corr_norms_2)
    std_2_norm_corr = np.std(delta_corr_norms_2)
    # Calculate the minimum between the Frobenius norm of the empirical 
    # covariance matrix difference and the 2-norm of the feature-targets 
    # correlations difference along with its mean and std
    min_delta_norm = np.minimum(delta_cov_norms_f[:,np.newaxis], delta_corr_norms_2)
    min_delta_norm_avg = np.mean(min_delta_norm)
    min_delta_norm_std = np.std(min_delta_norm)
    print("Average difference of Frobenuous norm", avg_f_norm_cov)
    print("Standard deviation of difference of Frobenius norm", std_f_norm_cov)
    print("Average difference of vector 2-norm", avg_2_norm_corr)
    print("Standard deviation of difference of vector 2-norm", std_2_norm_corr)
    print("Averange difference of min (F,2)", min_delta_norm_avg)
    print("Standard deviation of min (F,2)", min_delta_norm_std)


def sample_datasets_new_deprecated(num_samples, filenames, raw_partition_function, seed):
    
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
    
    def get_cumulative_partial_sums(results):
        # Extract partial sums from results
        partial_sums = [element[2] for element in results]
        cumulative_sums = np.cumsum(partial_sums)
        return cumulative_sums
    
    np.random.seed(seed)
    sample_indices = []
    
    # Sort filenames to ensure correct access of stored data
    filenames  = sorted(filenames, key = get_batch_id)
    
    # Scale partition function
    scaled_partitions = raw_partition_function * np.random.random(size=(num_samples,))
    print ("Scaled partitions with seed " + str(seed) + " are for the new algorithm are", scaled_partitions)
    # Obtain cumulative partition function - needs to be a numpy array in the actual implementation
    cumulative_partitions = get_cumulative_partial_sums(results)
    batches = np.searchsorted(cumulative_partitions, scaled_partitions)
    
    # If we were to move to the next matrix then we would get a negative score
    assert np.all(scaled_partitions - cumulative_partitions[batches] < 0.0)
    
    # Adjust indices output to get the index for which the partition is still positive
    #if not np.any(batches == 0):
    
    # Shrink partitions
    # TODO: So if batch = 0 the partition remains unchanged
    scaled_partitions[batches >= 1] = scaled_partitions[batches >= 1] - cumulative_partitions[batches[batches >= 1] - 1]
        
    # Expect to have positive scores
    assert np.all(scaled_partitions >= 0.0) 
    
    # Create a dictionary with each batch as a separate key, to handle cases when 
    # there are multiple samples from the same batch without loading the data twice
    batch_dictionary = {}
    
    for key,value in zip(batches, scaled_partitions):
        batch_dictionary.setdefault(key, []).append(value)
        
    # For each batch, load the data and calculate the row index
    for key in batch_dictionary.keys():
        scores = testutilities.retrieve_scores(filenames, batches= [key])[0]['scores']
        max_row_idx = scores.shape[0] - 1
        max_col_idx = scores.shape[1] - 1
        # Calculate the cumulative scores
        cum_scores = np.cumsum(np.sum(scores, axis=1))        
        
        # Find the rows in the score matrix
        row_indices = np.searchsorted(cum_scores, batch_dictionary[key])
        
        # Rescale partitions to accound for the contribution of rows
        partition_residuals = np.zeros(shape=(len(row_indices,)))
        partition_residuals[row_indices >= 1] = np.array(batch_dictionary[key])[row_indices >= 1] - cum_scores[row_indices[row_indices >= 1] - 1]
        if np.any(row_indices < 1):
            partition_residuals[row_indices < 1] = np.array(batch_dictionary[key])[row_indices < 1]
            
        # Determine the column index for each partition residual in the corresponding row
        col_indices = []
        for i in range(len(row_indices)):
            col_index = np.searchsorted(np.cumsum(scores[row_indices[i],:]), partition_residuals[i])
            if  col_index > 0:
                col_indices.append(col_index - 1)
            else:
                col_indices.append(max_col_idx)
                row_indices[i] = row_indices[i] - 1

        # Test that the column index calculation is correct
        for row_index, col_index, partition_residual in zip(row_indices, col_indices, partition_residuals):
            assert np.all(partition_residual - np.cumsum(scores[row_index,:])[col_index]) > 0
            if col_index < max_col_idx:
                assert np.all(partition_residual - np.cumsum(scores[row_index,:])[col_index + 1] < 0)
            else:
                assert np.all(partition_residual - np.cumsum(scores[row_index + 1][0]) < 0)
        # Add index tuples to the list
        for batch_idx, row_idx, col_idx in zip([key]*len(row_indices), row_indices, col_indices):
            if int(row_idx) == 0 and int(col_idx) == 0:
                sample_indices.append((batch_idx - 1, max_row_idx, max_col_idx,0))
            sample_indices.append((batch_idx, int(row_idx), int(col_idx),0))
    
    print ("Sampled indices returned by the new algorithm with seed " + str(seed) + " are", sample_indices)    
    return sample_indices

def sample_datasets_new(num_samples, filenames, raw_partition_function, seed):
    
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
    
    def get_cumulative_partial_sums(results):
        # Extract partial sums from results
        partial_sums = [element[2] for element in results]
        cumulative_sums = np.cumsum(partial_sums)
        return cumulative_sums
    
    np.random.seed(seed)
    sample_indices = []
    
    # Sort filenames to ensure correct access of stored data
    filenames  = sorted(filenames, key = get_batch_id)
    
    # Scale partition function
    scaled_partitions = raw_partition_function * np.random.random(size=(num_samples,))
    print ("Scaled partitions with seed " + str(seed) + " are for the new algorithm are", scaled_partitions)
    # Obtain cumulative partition function - needs to be a numpy array in the actual implementation
    cumulative_partitions = get_cumulative_partial_sums(results)
    batches = np.searchsorted(cumulative_partitions, scaled_partitions)
    
    # If we were to move to the next matrix then we would get a negative score
    assert np.all(scaled_partitions - cumulative_partitions[batches] < 0.0)
    
    # Adjust indices output to get the index for which the partition is still positive
    #if not np.any(batches == 0):
    
    # Shrink partitions
    # TODO: So if batch = 0 the partition remains unchanged
    scaled_partitions[batches >= 1] = scaled_partitions[batches >= 1] - cumulative_partitions[batches[batches >= 1] - 1]
        
    # Expect to have positive scores
    assert np.all(scaled_partitions >= 0.0) 
    
    # Create a dictionary with each batch as a separate key, to handle cases when 
    # there are multiple samples from the same batch without loading the data twice
    batch_dictionary = {}
    
    for key,value in zip(batches, scaled_partitions):
        batch_dictionary.setdefault(key, []).append(value)
        
    # For each batch, load the data and calculate the row index
    for key in batch_dictionary.keys():
        scores = testutilities.retrieve_scores(filenames, batches= [key])[0]['scores']
        max_col_idx = scores.shape[1] - 1
        # Calculate the cumulative scores
        cum_scores = np.cumsum(np.sum(scores, axis = 1))        
        
        # Find the rows in the score matrix
        row_indices = np.searchsorted(cum_scores, batch_dictionary[key])
        
        # Rescale partitions to accound for the contribution of rows
        partition_residuals = np.zeros(shape=(len(row_indices,)))
        partition_residuals[row_indices >= 1] = np.array(batch_dictionary[key])[row_indices >= 1] - cum_scores[row_indices[row_indices >= 1] - 1]
        if np.any(row_indices < 1):
            partition_residuals[row_indices < 1] = np.array(batch_dictionary[key])[row_indices < 1]
            
        # Determine the column index for each partition residual in the corresponding row
        col_indices = []
        for i in range(len(row_indices)):
            col_index = np.searchsorted(np.cumsum(scores[row_indices[i],:]), partition_residuals[i])
            if col_index > max_col_idx:
                col_indices.append(0)
                row_indices[i] = row_indices[i] + 1   
            else:
                col_indices.append(col_index)

        for row_index, col_index, partition_residual in zip(row_indices, col_indices, partition_residuals):
            assert np.all(partition_residual - np.cumsum(scores[row_index,:])[col_index] < 0 )
            if col_index > 0:
                assert np.all(partition_residual - np.cumsum(scores[row_index,:])[col_index - 1] > 0)
            #else:
            #    assert np.all(partition_residual - np.cumsum(scores[row_index - 1, :][max_col_idx]) > 0)
        # Add index tuples to the list
        for batch_idx, row_idx, col_idx in zip([key]*len(row_indices), row_indices, col_indices):
            # if int(row_idx) == 0 and int(col_idx) == 0:
            #    sample_indices.append((batch_idx - 1, max_row_idx, max_col_idx,0))
            sample_indices.append((batch_idx, int(row_idx), int(col_idx),0))
    
    print ("Sampled indices returned by the new algorithm with seed " + str(seed) + " are", sample_indices)    
    return sample_indices

def compare_sampled_indices(sample_1, sample_2, length = 4):
    ''' Compares if the sampled indices in the two tuple
    arrays are identical. Preprocessed to remove the fourth entry'''
    
    def remove_entry(tuple_array):
        return [tuple([el_1,el_2,el_3]) for el_1,el_2,el_3,el_4 in tuple_array]
    
    def sort_tuple_array(tuple_array):
        return sorted(tuple_array, key = lambda element: (element[0], element[1]))
    
    if length == 4:
        sample_1 = sort_tuple_array(remove_entry(sample_1))
        sample_2 = sort_tuple_array(remove_entry(sample_2))
    else:
        sample_1 = sort_tuple_array(sample_1)
        sample_2 = sort_tuple_array(sample_2)
    
    assert sample_1 == sample_2
    print ("Sampling indices are the same for the old and new algorithm")
    
# Storage
utility_arrays = [] # Utility arrays

# Declare experiment parameters
dim = 2
num_points_feat = 3
num_points_targ = 3
batch_size = 10
n_private = 40

# Declare privacy paramters
epsilon = 2.0
scaled_epsilon = epsilon/2 
# Declare synthetic space elements
OutputLattice = FeaturesLattice()
OutputLattice.generate_l2_lattice(dim = dim, num_points = num_points_feat)
features = OutputLattice.points
OutputLattice2 = TargetsLattice()
OutputLattice2.generate_lattice(dim = dim, num_points = num_points_targ)
targets = OutputLattice2.points

# Generate synthethic data and compute its utility
private_data = ContinuousGenerator(d = dim, n = n_private)
private_data.generate_data(test_frac = 0.5)
print ("Coefficients of the model from which the private data was generated are", private_data.coefs)
F_tilde_x = testutilities.get_private_F_tilde(private_data)

# Inverse global sensitivity
igs = private_data.features.shape[0]/2 

# Utility scaling constant 
scaling_const = igs*scaled_epsilon

# Calculate number of batches
n_batches = math.ceil(comb(features.shape[0], dim, exact = True)/batch_size)
print ("Number of batches is", n_batches)
# Define directory where the files will be saved
experiment_name = 'test_struct_integrity'
directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/' + experiment_name + '/OutcomeSpace'

# Define file name for scores (s) storage
base_filename_s = "s_eps" + str(epsilon).replace(".", "") + "d" + str(dim)

t_start = time.time()
results = []
for batch_index in range(n_batches):
    results.append(testutilities.evaluate_sample_score(batch_index, features, targets, scaling_const, F_tilde_x, dim, batch_size, \
                                                       base_filename_s, directory))
t_elapsed = time.time()


print("Time elapsed for single core processing of this small case is..." + " " + str(t_elapsed - t_start))


res = np.exp(results[0][1]).flatten().tolist()
res_freqs = collections.Counter(res)
# Test that data reloading function works - seems to work fine

if directory.rfind("*") == -1:
    directory = directory + "/*"

# We pass filenames to the data loader to avoid calling glob.glob for every sampling step
filenames = glob.glob(directory)

reloaded_data = testutilities.retrieve_scores(filenames)

is_diff = []
rtol = 1e-05

for reloaded_element,returned_element in zip(reloaded_data,results):
    difference = reloaded_element['scores'] - returned_element[1]
    is_diff.append(np.all(np.isclose(difference,np.zeros(shape=difference.shape),rtol=rtol)))

assert np.all(is_diff)

# Test reloading of only some batches - seems to work fine!

#batches = [0,1,2]
#
#reloaded_data_batches = testutilities.retrieve_scores(filenames,batches)
#
#is_diff = []
#rtol = 1e-05
#
#sliced_results = [results[x] for x in batches]
#
#for reloaded_element, returned_element in zip(reloaded_data_batches,sliced_results):
#    difference = reloaded_element['scores'] - returned_element[1]
#    is_diff.append(np.all(np.isclose(difference,np.zeros(shape=difference.shape),rtol=rtol)))
#
#assert np.all(is_diff)
 
# Calculating the partition function...

if directory.rfind("*") == -1:
    directory = directory + "/*"

# We pass filenames to the data loader to avoid calling glob.glob for every sampling step
filenames = glob.glob(directory)

# Calculate max_score for sum-exp trick
max_scaled_utility = - math.inf

for result in results:
    if result[0] > max_scaled_utility:
        max_scaled_utility = result[0]
 
print ("Max scaled utility, simple method", max_scaled_utility)
print ("Max utility, simple method", 1/scaling_const* max_scaled_utility)

# Calculate partition function (without sum_exp)

raw_partition_function = 0
debug_partition_function = 0

for result in results:
    raw_partition_function += np.sum(np.exp(result[1]))
    
intermediate_partition_values = []    
    
for result in results:
    intermediate_partition_values.append(np.sum(np.exp(result[1] - max_scaled_utility)))
    debug_partition_function += np.sum(np.exp(result[1] - max_scaled_utility))
    
print ("Raw partition value", raw_partition_function)
print ("Debug partition function value", debug_partition_function)

# Calculate partition function (using sum_exp) 
partition_function = calculate_partition_function(filenames, n_batches, test = False, max_score = 0.0)

print ("Sum-exp partition value",partition_function)

filenames_orig = [filename for filename in glob.glob(directory) if filename.find("orig_") >= 0]

# Check that the original data has not been corrupted when written to file...

#for batch in range(n_batches):
#
#    # Load the original file and perform the computation on it
#    
#    data_orig = retrieve_scores(filenames_orig,batches=[batch])[0]
#    
#    # Compare data_orig with the result returned by the function 
#    
#    differece = data_orig['scores'] - results[batch][1]
#    exp_diff = np.zeros(shape=difference.shape)
#    
#    assert np.all(np.isclose(difference,exp_diff,rtol=rtol))

# Check that the combinations have been correctly preserved and that the data has been
# manipulated correctly
    
#for batch in range(n_batches):
#    
#    # Load original and modified data files
#    
#    data_modified = retrieve_scores(filenames,[batch])[0] 
#    data_orig = retrieve_scores(filenames_orig,batches=[batch])[0]
#    
#    # Check the combinations field is idenfical
#    assert list(data_modified['combs']) == list(data_orig['combs'])
#    
#    exp_val = np.exp(data_orig['scores'] - max_score)
#    difference = exp_val - data_modified['scores']
#   # assert np.all(np.isclose(difference,np.zeros(shape=difference.shape),rtol=rtol))
   
partition_function_alt = calculate_partition_function_alt(iter(results))

print ("Alternative method for calculating partition function gives", partition_function_alt)

# Now let's test the sampling procedure...

num_samples = 25
seed = 23

sample_indices = sample_dataset(n_batches, num_samples, raw_partition_function, filenames, seed)

# Use another method to calculate partition function given the output of the sampling function
            
partition_residuals = testutilities.check_sampling(sample_indices, results, max_score = 0.0)
print ("Partition residuals for the old method with seed " + str(seed) + " are:", partition_residuals)

# Modify column indices and calculate residuals - they should be negative

list_conversion = [list(element) for element in sample_indices]
# deprecated: sample_indices_modified = [[batch_index, row_index, col_index + 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]
sample_indices_modified = [[batch_index, row_index, col_index - 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]
#
partition_mod_residuals = testutilities.check_sampling(sample_indices_modified, results, max_score = 0.0)    

print ("Partition residuals for the old method with seed" + str(seed) + "are:", partition_mod_residuals)

# depracted mask = [True if element <= 0 else False for element in partition_mod_residuals]

mask = [True if element >= 0.0 else False for element in partition_mod_residuals]

assert (all(mask))

# And finally the matrix recovery procedure...   

synthetic_data_sets = np.array(testutilities.recover_synthetic_datasets(sample_indices, features, targets, batch_size, dim ))
synthetic_data_sets_alternative =  np.array(testutilities.recover_synthetic_datasets(sample_indices, features, targets, batch_size, dim))
#print ("The data sets sampled are", synthetic_data_sets)

# Recalculate scores and utilities based on the recovered synthetic data sets
calculated_scores, calculated_scaled_utilities, _ = testutilities.calculate_recovered_scores(synthetic_data_sets, F_tilde_x, scaling_const, dim)

samples_utilities =  np.array((1/scaling_const)*calculated_scaled_utilities)

print("Samples utilities", samples_utilities)
print("Samples scaled utilities", calculated_scaled_utilities)
print("Samples scores", calculated_scores)
print("Samples utilities (average)", np.mean(samples_utilities))
print("Samples utilties (standard dev)", np.std(samples_utilities))

samples_utilities_freqs = collections.Counter(samples_utilities)
calculated_scaled_utilities_freqs = collections.Counter(calculated_scaled_utilities)
calculated_scores_freqs = collections.Counter(calculated_scores)

# Retrieve the scores from the raw results
# This assumes the results contain RAW results (aka exp not taken)
look_up_scores = testutilities.retrieve_scores_from_results(results, sample_indices, max_scaled_utility = 0.0)        

# Compare retrieved and calculated scores
assert  np.all(np.isclose(np.array(calculated_scores), np.array(look_up_scores), rtol = rtol))

# New sampling methodology test
seed = 23
new_sample_indices = sample_datasets_new(num_samples, filenames, raw_partition_function, seed)

new_partition_residuals = testutilities.check_sampling(new_sample_indices, results,max_score = 0.0)

print ("Partition subtracted for the new algorithm is", new_partition_residuals)

# Compare sample indices 
compare_sampled_indices(sample_indices, new_sample_indices)


list_conversion = [list(element) for element in new_sample_indices]
# deprecatedd: new_sample_indices_modified = [[batch_index, row_index, col_index + 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]

new_sample_indices_modified = [[batch_index, row_index, col_index - 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]
new_partition_check = testutilities.check_sampling(new_sample_indices_modified, results,max_score = 0.0)

print ("Partition subtracted for the new algorithm is (check)", new_partition_check)

# Calculate accuraccies
print("Accuracies of current sampling implementation")
calculate_accuracy(synthetic_data_sets, dim, F_tilde_x)
print("Accuracies of alternative sampling implementation")
calculate_accuracy(synthetic_data_sets_alternative, dim, F_tilde_x)

# Check regression results
netmech_regressor = Regression()
# regressor_alternative = Regression()

param = netmech_regressor.fit_data(synthetic_data_sets)
predictive_err_netmech = netmech_regressor.calculate_predictive_error(private_data.test_data, param)
#print ("Predictive_errors for net mechanism", predictive_err_netmech)
print ("Min predictive error net mechanism", np.min(predictive_err_netmech))
print ("Mean predictive error net mechanism", np.mean(predictive_err_netmech))
print ("Std of predictive err net mechanism", np.std(predictive_err_netmech))

# Fit pamaters with ADASSP algorithm

adassp_regressor = DPRegression()
adassp_reg_coef = adassp_regressor.get_parameters(private_data.features, private_data.targets, num_samples, epsilon)
predictive_err_adassp = Regression().calculate_predictive_error(private_data.test_data, adassp_reg_coef)
# print ("Predictive_errors for adassp", predictive_err_adassp)
print ("Min predictive error adassp", np.min(predictive_err_adassp))
print ("Mean predictive error adassp", np.mean(predictive_err_adassp))
print ("Std of predictive err adassp", np.std(predictive_err_adassp))


# Calculate Frobenius norm of the covariance difference  and the 2-norm of the correlations difference
emp_cov_synth = 1/dim * np.transpose(synthetic_data_sets[:,:,:-1], axes = (0,2,1))@synthetic_data_sets[:,:,:-1]
emp_corr = 1/dim * np.transpose(synthetic_data_sets[:,:,:-1], axes = (0,2,1))@synthetic_data_sets[:,:,-1:]

delta_cov = F_tilde_x[:,:-1] - emp_cov_synth
delta_corr = F_tilde_x[:,-1:] - emp_corr

delta_cov_norms_f = np.linalg.norm(delta_cov, ord = 'fro', axis = (1,2))
delta_cov_norms_2 = np.linalg.norm(delta_cov, ord = 2, axis = (1,2))
delta_corr_norms_2 = np.linalg.norm(delta_corr, ord = 2, axis = 1)

avg_f_norm_cov = np.mean(delta_cov_norms_f)
std_f_norm_cov = np.std(delta_cov_norms_f)
avg_2_norm_corr = np.mean(delta_corr_norms_2)
std_2_norm_corr = np.std(delta_corr_norms_2)
# param_alternative = regressor_alternative.fit_data(synthetic_data_sets_alternative)
# param_alternative_2 = regressor_alternative2.fit_data(synthetic_data_sets_alternative_2)

# To peform these tests, run second_moment_experiments_main.py with the same parameters

## assert  np.all(np.isclose(synthetic_data_integrated,synthetic_data_sets,rtol = rtol))
##%% # Test synthetic data saving
##import testutilities
##
##path = 'D:/Thesis/Experiments/s_eps01d2nt5nf8/SyntheticData/s_eps01d2nt5nf8'
##
##data = testutilities.load_data(path)
##%%
## Investigate empty batches problem...
#import testutilities, glob
#
#def get_batch_id(filename):
#    return int(filename[filename.rfind("_") + 1:])
#
#
#
#experiment_name = 's_eps01d3nt10nf10'
#exp_directory = 'D:/Thesis/Experiments/' + experiment_name + '/OutcomeSpace'
#filenames = glob.glob(exp_directory + "/*")
#filenames  = sorted(filenames, key = get_batch_id)
#
#dir1= 'D:/Thesis/Experiments/s_eps01d3nt10nf10/OutcomeSpace/'
#dir2 = 'D:/Thesis/Experiments/s_eps01d3nt10nf10_shutdown/OutcomeSpace/'
#
#fname1 = 's_eps01d3nt10nf10_0'
#fname2 = 's_eps01d3nt10nf10_5'
#
## Processed batches
#
## Does the data in batch 0 correspond in both experiment
#
#loaded_data_1_1 = testutilities.retrieve_scores([dir1+fname1])
#loaded_data_2_1 = testutilities.retrieve_scores([dir2+fname1])
# batch = 3
 #data = testutilities.retrieve_scores(filenames,[batch])
#%% Testing noise addition to generated data
#from data_generators import ContinuousGenerator 
## Generate noisy data
#generator = ContinuousGenerator(d = 1, n = 10, perturbation = True, perturbation_mean = 0, perturbation_variance= 0.01)
## Generate data
#generator.generate_data(seed = 23, test_frac = 0.1)
#dataset = generator.data

#%% Testing Regression class ()
#from baselines import Regression
#import pickle
#
#path_1  = 'D:/Thesis/Experiments/s_eps01d2nt6nf6/SyntheticData/s_eps01d2nt6nf6'
#path_2  = 'D:/Thesis/Experiments/s_eps01d2nt20nf20/SyntheticData/s_eps01d2nt20nf20'
#with open(path_1, mode = 'rb') as container:
#    data_1 = pickle.load(container)
#
#with open(path_2, mode = 'rb') as container:
#    data_2 = pickle.load(container)
#
#regressor_1 = Regression()
##parameters_1 = regressor_1.fit_data(data_1)
#regressor_2 = Regression()
#parameters_2 = regressor_2.fit_data(data_2)
#%% Testing experimental setup
#import second_moment_experiments_main as experiment
#from exputils import extract_data, initialise_netmech_containers 
#from baselines import Regression, DPRegression
#import numpy as np
#
## Default parameters list
#dimensionality = 2
#num_records = 40
#test_frac = 0.5
#batch_size = 100
#directory = 'D:/Thesis/Experiments/exp_2/'
#parallel = False
#save_data = False
#partition_method = 'fast_2'
#workers = -1
#num_samples = 25
#sample_parallel = False 
#load_data = False
#seed = 23
#
#num_points_max = 20
#num_points_min = 4
#num_points_features_vec = range(num_points_min, num_points_max + 1)
#num_points_targets_vec = range(num_points_min, num_points_max + 1)
#
#epsilon_vec = [0.1, 1.0]
#results = {key: [] for key in epsilon_vec}
#
## Collect results
#for epsilon in epsilon_vec:
#    for num_points_features, num_points_targets in zip(num_points_features_vec, num_points_targets_vec):
#        results[epsilon].append(experiment.second_order_moment_experiment(dimensionality = dimensionality, num_records = num_records, test_frac = test_frac, batch_size = batch_size,directory = directory, parallel = parallel, save_data = save_data,\
#                                                                 partition_method = partition_method, workers = workers, num_samples = num_samples,\
#                                                                 sample_parallel = sample_parallel, load_data = load_data, num_points_targets = num_points_targets,\
#                                                                 num_points_features = num_points_features, epsilon = epsilon, seed = seed))
#
## Experimental data containers
#avg_2_norms, double_std_2_norms, avg_f_norms, double_std_f_norms, max_utilities, max_sampled_utilities, avg_samples_utility,\
#double_std_utility, avg_samples_score, double_std_score, synthetic_datasets_vec, test_set, private_data = initialise_netmech_containers(epsilon_vec)
#
#for key in results:
#    avg_2_norms[key], double_std_2_norms[key], avg_f_norms[key], double_std_f_norms[key], max_utilities[key], max_sampled_utilities[key],\
#    avg_samples_utility[key], double_std_utility[key], avg_samples_score[key], double_std_score[key], synthetic_datasets_vec[key],\
#    test_set[key], private_data[key] = extract_data(results[key]) 
#
#adassp_reg_coef = {key: [] for key in epsilon_vec}
#predictive_err_adassp = {key: [] for key in epsilon_vec}
#min_predictive_err_adassp = {key: [] for key in epsilon_vec}
#mean_predictive_err_adassp = {key: [] for key in epsilon_vec}
#double_std_predictive_err_adassp = {key: [] for key in epsilon_vec}
#
## Fit ADASSP to the private dataset
#for epsilon in epsilon_vec:
#    adassp_regressor = DPRegression()
#    adassp_reg_coef[epsilon] = adassp_regressor.get_parameters(private_data[epsilon].features, private_data[epsilon].targets,\
#                                                           num_samples, epsilon)
#    predictive_err_adassp[epsilon] = Regression().calculate_predictive_error(private_data[epsilon].test_data, adassp_reg_coef[epsilon])
#    min_predictive_err_adassp[epsilon] = np.min(predictive_err_adassp[epsilon])
#    mean_predictive_err_adassp[epsilon] = np.mean(predictive_err_adassp[epsilon])
#    double_std_predictive_err_adassp[epsilon] = 2*np.std(predictive_err_adassp[epsilon])
#    print ("Min predictive error adassp for eps " + str(epsilon) , min_predictive_err_adassp[epsilon])
#    print ("Mean predictive error adassp " + str(epsilon), mean_predictive_err_adassp[epsilon])
#    print ("Twice the std of predictive err adassp " + str(epsilon), double_std_predictive_err_adassp[epsilon])
#    
#net_mech_reg_coefs = {key: [] for key in epsilon_vec}
#predictive_errs_netmech = {key: [] for key in epsilon_vec}
#min_predictive_errs_netmech = {key: [] for key in epsilon_vec}
#mean_predictive_errs_netmech = {key: [] for key in epsilon_vec}
#double_std_predictive_errs_netmech = {key: [] for key in epsilon_vec}
#
#for epsilon in epsilon_vec: 
#    for synthetic_datasets in synthetic_datasets_vec[epsilon]:
#        netmech_regressor = Regression()
#        net_mech_reg_coef = netmech_regressor.fit_data(synthetic_datasets)
#        net_mech_reg_coefs[epsilon].append(net_mech_reg_coef)
#        predictive_err_netmech = netmech_regressor.calculate_predictive_error(private_data[epsilon].test_data, net_mech_reg_coef)
#        predictive_errs_netmech[epsilon].append(predictive_err_netmech)
#        min_predictive_errs_netmech[epsilon].append(np.min(predictive_err_netmech))
#        mean_predictive_errs_netmech[epsilon].append(np.mean(predictive_err_netmech))
#        double_std_predictive_errs_netmech[epsilon].append(2*np.std(predictive_err_netmech))
#    print("Overall minimum predictive error for netmechanism " + str(epsilon) + " is {}, obtained for n_t = {}."\
#          .format(str(np.min(np.array(min_predictive_errs_netmech[epsilon]))),\
#                  str(list(num_points_features_vec)[np.argmin(np.array(min_predictive_errs_netmech[epsilon]))])))
#    print("Minimum average predictive error for netmechanism " + str(epsilon) + " is {}, obtained for n_t = {}."\
#          .format(str(np.min(np.array(mean_predictive_errs_netmech[epsilon]))),\
#                  str(list(num_points_features_vec)[np.argmin(np.array(mean_predictive_errs_netmech[epsilon]))])))
#%% 
# Plot the dataset 
from data_generators import ContinuousGenerator
import numpy as np
dimensionality = 2
num_records = 40
test_frac = 0.5
private_data = ContinuousGenerator(d = dimensionality, n = num_records)
private_data.generate_data(test_frac = test_frac)   
# private_data.plot_data()

test_features = private_data.test_features
test_features_norms = np.linalg.norm(test_features, ord = 2, axis = 1)
train_data = private_data.data

