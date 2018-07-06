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

dim = 10
num_points = 3
upper_bound = 1.0
lower_bound = -1.0
num_dec = 4
radius = 1.0
r_tol = 1e-5
OutputLattice = FeaturesLattice()
OutputLattice.generate_l2_lattice(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points,pos_ord=True,rel_tol=r_tol)
intersection_m2 = testutilities.bruteNonIntegerIntersection(dim=dim,radius=radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound,filtered = False,r_tol=r_tol)
test_points = OutputLattice.points
## Points that are returned by the fancy algorithm but not by brute
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


def get_private_F_tilde (private_data):
    
    # Compute F_tilde (equation (4.1), Chapter 4, Section 4.1.1)
    # for the private data
    
    const = (1/private_data.features.shape[0])
    F_x = const*private_data.features.T@private_data.features
    f_x = const*private_data.features.T@private_data.targets
    F_tilde_x = np.concatenate((F_x,f_x), axis = 1)
    
    return F_tilde_x        
    
# @profile

def save_batch_scores(batch, filename, directory = '', overwrite = False):
    ''' Saves the batch of scores to the location specified by @directory with
    the name specified by @filename.'''
    
    # If directory is not specified, then the full path is provided in filename
    # This is used to overwrite the files containing the raw scores during the 
    # calculation of the partition function to avoid unnecessary duplication of 
    # operations during the sampling step
    
    if not directory:
        full_path = filename
        
        if overwrite:
            with open(full_path,"wb") as data:
                pickle.dump(batch,data)
        else:
            # Overwriting data only alowed if this is explicitly mentioned
            if os.path.exists(full_path):
                assert False
            
    else:
        
        # Create directories if they don't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        full_path = directory+"/"+filename
       
        # Raise an error if the target file exists
#        if os.path.exists(full_path):
#                assert False
           
        with open(full_path,"wb") as data:
            pickle.dump(batch,data)

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
    
    print ("Max score evaluated inside calculate_partition_function", max_score)
    
    partition_function  = 0
    
    for batch in range(n_batches):
        data = testutilities.retrieve_scores(filenames,[batch])[0]
        
        if test:
            # Keep a copy of the original data to allow testing 
            fname = "orig_" + filenames[batch][filenames[batch].rfind("\\") + 1:]
            save_batch_scores(data,fname,directory.replace("*",""))
            
        # Apply sum-exp trick when calc. partition to avod numerical issues
        data['scores'] = np.exp(data['scores'] - max_score)
        partition_function += np.sum(data['scores'])
        
        # Overwrite the raw batch scores with the scores calculated as per 
        # Step 3 of the procedudre in Chapter 4, Section 4.1.3 
        save_batch_scores(data, filenames[batch], overwrite = True)
    
    return partition_function

#@profile
def compute_second_moment_utility(outcomes):
                         
    f_r_tensor = (1/dim)*np.matmul(targets,outcomes)
    
    # Calculate F_r = 1/d Xh'Xh (' denotes transpose). This is applied for all Xh in the synth_features_tensor
    F_r_tensor = (1/dim)*np.transpose(outcomes,axes=(0,2,1))@outcomes
    
    #TODO: add comment
    f_r_expand = f_r_tensor.reshape(tuple([*f_r_tensor.shape,1]))
    
    #TODO: add comment
    F_r_expand = np.repeat(F_r_tensor,repeats=targets.shape[0],axis=0).reshape(F_r_tensor.shape[0],-1,*F_r_tensor[0].shape)
    
    #TODO: add comment
    F_tilde_r = np.concatenate((F_r_expand,f_r_expand),axis=3)
    
    # Utilities for the particular batch are returned as a matrix of dimension batch_size x p where p is the number of 
    # synthetic targets. Exp-normalise trick is implemented so the exponentiation is done in the sampling step
    utility = - scaling_const*np.max(np.abs(F_tilde_x-F_tilde_r),axis=(3,2))
    
    return utility

# @profile
def evaluate_sample_score(batch_index,test = False):
    
    # Storage structure
    struct = {}
    
    # Store the batch index to be able to retrieve the correct sample during sampling step
    struct['batch_index']  = batch_index
    
    # Generate a batch of combinations according to the batch_index
    batch = list(itertools.islice(itertools.combinations(range(features.shape[0]),dim),(batch_index)*batch_size,(batch_index+1)*batch_size))
    
    # Evaluate utility - note that exponential is not taken as sum-exp trick is implemented to 
    # evalute the scores in a numerically stable way during sampling stage
    score_batch = compute_second_moment_utility(features[batch,:])
    struct ['scores'] = score_batch
    struct ['test_data'] = batch
    
    # Create data structure which stores the scores for each batch along with 
    # the combinations that generated them
    max_util = np.max(score_batch)
    
    # save the slice object
    filename = "/" + base_filename_s + "_" + str(batch_index)
    save_batch_scores(struct,filename,directory)
    
    partial_sum = np.sum(np.exp(score_batch))
    # Only max_util is returned in the final version of the code to 
    # allow implementation of exp-normalise trick during sampling . 
    # score_batch is returned for testing purposes
    
    return (max_util,score_batch, partial_sum)
    
def calculate_partition_function_alt(iterable):
    
    def func(scores_batch,max_score):
        return np.sum(np.exp(scores_batch[1]-max_score))
        
    iter_1,iter_2 = itertools.tee(iterable)
    
    ms = functools.reduce((lambda x,y:(max(x[0],y[0]),np.zeros(shape=y[1].shape))),iter_1)[0]    
    
    print ("Max score evaluated by the alternative partition function calculation",ms)
    
    partition_function = sum(map(functools.partial(func,max_score=ms),iter_2))
    
    return partition_function

def sample_dataset(n_batches, num_samples, partition_function, filenames, seed):
    
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
                    print ("The value of the score is", element)
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

def recover_synthetic_datasets(sample_indices):
    
    def nth(iterable, n, default=None):
        "Returns the nth item from iterable or a default value"
        return next(itertools.islice(iterable, n, None), default)
    
    # Data containers
    feature_matrices = []
    synthetic_data_sets = []
    
    # Batches the samples were drawn from
    batches = [element[0] for element in sample_indices]   
    
    # Combinations corresponding to the element which resulted in the zero crossing
    # These are used to recover the feature matrices
    combs_idxs = [element[1] for element in sample_indices]
    
    # List of indices of the target vectors for the sampled data sets
    target_indices = [element[2] for element in sample_indices]
    
    # Feature matrix reconstruction 
    for batch_idx, comb_idx in zip(batches, combs_idxs):
        
        # Reconstruct slice
        recovered_slice = itertools.islice(itertools.combinations(range(features.shape[0]), dim), (batch_idx)*batch_size, (batch_idx+1)*batch_size)
        
        # Recover the correct combination 
        combination = nth(recovered_slice, comb_idx)
        print ("Recovered combination", combination)
    
        # Recover the feature matrix
        feature_matrices.append(features[combination,:])

    # Reconstruct the targets for the synthethic feature matrix 
    for feature_matrix,target_index in zip(feature_matrices,target_indices):
        synthetic_data_sets.append(np.concatenate((feature_matrix, targets[target_index,:].reshape(targets.shape[1],1)), axis = 1))
        
    return synthetic_data_sets 

# Storage
utility_arrays = [] # Utility arrays

# Declare experiment parameters
dim = 2
num_points_feat = 8
num_points_targ = 5
batch_size = 500
n_private = 20

# Declare privacy paramters
epsilon = 0.1
scaled_epsilon = epsilon/2 

# Declare synthetic space elements
OutputLattice = FeaturesLattice()
OutputLattice.generate_l2_lattice(dim = dim,num_points = num_points_feat)
features = OutputLattice.points
OutputLattice2 = TargetsLattice()
OutputLattice2.generate_lattice(dim = dim,num_points = num_points_targ)
targets = OutputLattice2.points

# Generate synthethic data and compute its utility
private_data = ContinuousGenerator(d = dim, n = n_private)
private_data.generate_data()
F_tilde_x = get_private_F_tilde(private_data)

# Inverse global sensitivity
igs = private_data.features.shape[0]/2 

# Utility scaling constant 
scaling_const = igs*scaled_epsilon

# Calculate number of batches
n_batches = math.ceil(comb(features.shape[0],dim,exact = True)/batch_size)

# Define directory where the files will be saved
experiment_name = 'test_struct_integrity'
directory = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Experiments/' + experiment_name + '/OutcomeSpace'

# Define file name for scores (s) storage
base_filename_s = "s_eps" + str(epsilon).replace(".","") + "d" + str(dim)

t_start = time.time()
results = []
for batch_index in range(n_batches):
    results.append(evaluate_sample_score(batch_index))
t_elapsed = time.time()

print("Time elapsed for single core processing of this small case is..." + " " + str(t_elapsed-t_start))

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
max_score = - math.inf

for result in results:
    if result[0] > max_score:
        max_score = result[0]
 
print ("Max score, simple method",max_score)

# Calculate partition function (without sum_exp)

raw_partition_function = 0
debug_partition_function = 0

for result in results:
    raw_partition_function += np.sum(np.exp(result[1]))
    
intermediate_partition_values = []    
    
for result in results:
    intermediate_partition_values.append(np.sum(np.exp(result[1] - max_score)))
    debug_partition_function += np.sum(np.exp(result[1] - max_score))
    
print ("Raw partition value",raw_partition_function)
print ("Debug partition function value",debug_partition_function)

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

num_samples = 50
seed = 23

sample_indices = sample_dataset(n_batches, num_samples, raw_partition_function, filenames, seed)

# Use another method to calculate partition function given the output of the sampling function
            
partition_residuals = testutilities.check_sampling(sample_indices, results, max_score = 0.0)
print ("Partition residuals for the old method with seed " + str(seed) + " are:",partition_residuals)

# Modify column indices and calculate residuals - they should be negative

list_conversion = [list(element) for element in sample_indices]
sample_indices_modified = [[batch_index, row_index, col_index + 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]

partition_neg_residuals = testutilities.check_sampling(sample_indices_modified, results, max_score = 0.0)    

print ("Partition residuals for the old method with seed" + str(seed) + "are:", partition_neg_residuals)

mask = [True if element <= 0 else False for element in partition_neg_residuals]

# assert (all(mask))

# And finally the matrix recovery procedure...   

synthetic_data_sets = recover_synthetic_datasets(sample_indices)

print ("The data sets sampled are", synthetic_data_sets)

# Recalculate scores and utilities based on the recovered synthetic data sets
calculated_scores, calculated_scaled_utilities = testutilities.calculate_recovered_scores(synthetic_data_sets, F_tilde_x, scaling_const, dim)

samples_utilities = - (1/scaling_const)*calculated_scaled_utilities

print("Samples utilities",samples_utilities)

# Retrieve the scores from the raw results
look_up_scores = testutilities.retrieve_scores_from_results(results, sample_indices, max_scaled_utility = 0.0)        

# Compare retrieved and calculated scores
assert  np.all(np.isclose(np.array(calculated_scores),np.array(look_up_scores),rtol = rtol))

# New sampling methodology test

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
    orig_partitions = scaled_partitions
    print ("Scaled partitions with seed " + str(seed) + " are for the new algorithm are", scaled_partitions)
    # Obtain cumulative partition function - needs to be a numpy array in the actual implementation
    cumulative_partitions = get_cumulative_partial_sums(results)
    batches = np.searchsorted(cumulative_partitions, scaled_partitions).flatten()
    
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
            if int(row_idx == 0) and int(col_idx) == 0:
                sample_indices(batch_idx - 1, max_row_idx, max_col_idx)
            sample_indices.append((batch_idx, int(row_idx), int(col_idx),0))
    
    print ("Sampled indices returned by the new algorithm with seed " + str(seed) + " are", sample_indices)    
    return (sample_indices,orig_partitions)

seed = 23
new_sample_indices,orig_partitions = sample_datasets_new(num_samples, filenames, raw_partition_function, seed)

new_partition_residuals = testutilities.check_sampling(new_sample_indices, results,max_score = 0.0)

print ("Partition subtracted for the new algorithm is", new_partition_residuals)

list_conversion = [list(element) for element in new_sample_indices]
new_sample_indices_modified = [[batch_index, row_index, col_index + 1, part_function] for batch_index, row_index, col_index, part_function in list_conversion]

new_partition_check = testutilities.check_sampling(new_sample_indices_modified, results,max_score = 0.0)

print ("Partition subtracted for the new algorithm is (check)", new_partition_check)

# To peform these tests, run second_moment_experiments_main.py with the same parameters

# assert  np.all(np.isclose(synthetic_data_integrated,synthetic_data_sets,rtol = rtol))
#%% # Test synthetic data saving
#import testutilities
#
#path = 'D:/Thesis/Experiments/s_eps01d2nt5nf8/SyntheticData/s_eps01d2nt5nf8'
#
#data = testutilities.load_data(path)
#%%
# Investigate empty batches problem...
import testutilities, glob

def get_batch_id(filename):
    return int(filename[filename.rfind("_") + 1:])



experiment_name = 's_eps01d3nt10nf10'
exp_directory = 'D:/Thesis/Experiments/' + experiment_name + '/OutcomeSpace'
filenames = glob.glob(exp_directory + "/*")
filenames  = sorted(filenames, key = get_batch_id)

dir1= 'D:/Thesis/Experiments/s_eps01d3nt10nf10/OutcomeSpace/'
dir2 = 'D:/Thesis/Experiments/s_eps01d3nt10nf10_shutdown/OutcomeSpace/'

fname1 = 's_eps01d3nt10nf10_0'
fname2 = 's_eps01d3nt10nf10_5'

# Processed batches

# Does the data in batch 0 correspond in both experiment

loaded_data_1_1 = testutilities.retrieve_scores([dir1+fname1])
loaded_data_2_1 = testutilities.retrieve_scores([dir2+fname1])
# batch = 3
 #data = testutilities.retrieve_scores(filenames,[batch])
