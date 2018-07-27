# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:03:44 2018

@author: alexc
"""
import numpy as np
import operator
import itertools
import math
import glob
import pickle
import os
from itertools import chain

def bruteNonIntegerIntersection(dim, radius, num_points = 5, lower_bound = -1, upper_bound = 1,filtered = False, r_tol = 1e-06):
    """ Generate a lattice inside the d-dimensional hypersphere. Brute force method,
    all coordinates are first generated on the d-dimensional hypercube and those outside the 
    hypershere discarded. 
    TODO: Params with explanations"""
    
    def filter_unsorted(array):
        """ Removes rows that are not sorted from a numpy array, @array """
        def is_sorted(iterable, reverse = True):
            """Check if the iterable is sorted, possibly reverse sorted."""
    
            def pairwise(iterable):
                """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
                a, b = itertools.tee(iterable)
                next(b, None)
                return zip(a, b)
        
            my_operator = operator.ge if reverse else operator.le
        
            return all(my_operator(current_element, next_element)
                           for current_element, next_element in pairwise(iterable))
        
        # Mask used for slicing array
        mask = []
        filtered_array = []
    
        # Construct mask
        for index in range(array.shape[0]):
            mask.append(is_sorted(array[index,:]))
        
        # Slice to find filtered array
        filtered_array = array[mask]
        
        return filtered_array  

    def filter_signed(array):
        """Removes rows for which there is at least one negative element from a numpy array, @ array"""
        def has_negative(iterable):
            is_positive = False
            for entry in iterable:
                if entry < 0.0:
                    is_positive = False
                    break
                else:
                    is_positive = True
            return is_positive 
                    
        # Mask used for slicing array
        mask = []
        filtered_array =[] 
        
        # Construct mask
        for index in range(array.shape[0]):
            mask.append(has_negative(array[index,:]))
        
        # Slice to find filtered array
        filtered_array = array[mask]
        
        return filtered_array
    
    # Define lattice range
    coord_array = list(np.linspace(lower_bound,upper_bound,num = num_points, endpoint = True))
    lattice_coordinates = [coord_array for i in range(dim)]
    
    # Generate vectors on the hypercube
    try:
        flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    except MemoryError:
        print ("Ran out of memory for density", num_points)
        return 0
        
        
    # Generate lattice
    norms = np.linalg.norm(flat_grid, ord = 2, axis = 1)
    close_norms = [True if math.isclose(np.linalg.norm(x),radius,rel_tol=r_tol) == True else False for x in norms]
    small_norms = [True if x <= radius else False for x in norms]
    indices = [x or y for x,y in zip(small_norms,close_norms)]
    intersection = flat_grid[indices]
    
    # Remove unordered and signed solutions
    if filtered:
        tmp = filter_unsorted(intersection)
        intersection = filter_signed(tmp)
    
    # Sort according to the first column 
    intersection = intersection[intersection[:,0].argsort()[::-1]]
        
    return intersection  

def get_differences_deprecated(list_a=[],list_b=[]):
    ''' Returns those elements that are in list_a but not in list_b'''
    #TODO: do this without casting to list, it's not efficient
    #TODO: This solution is not robust - does not correctly check the negative
    # numbers
    if isinstance(list_a,np.ndarray):    
        list_a = [list(element) for element in list_a]
    if isinstance(list_b,np.ndarray):
        list_b = [list(element) for element in list_a]
    return [x for x in list_a if x not in list_b]

def get_differences(array_a=[],array_b=[],rel_tol=1e-5):
    ''' Returns the vectors that are in array_a but not in array_b'''
    def in_array(array,rel_tol):
        # TODO: Find out why the function fails with math.isclose instead of np.isclose
        ''' Checks whether the vector @candidate is one of the rows of the matrix @array'''
        for index in range(array.shape[0]):
            indicator = np.all([True if np.isclose(a,0.0,rtol=rel_tol) == True else False for a in array[index,:]])
            if indicator == True:
                break
            else:
                continue
        return indicator
    
    # Convert to numpy array
    if isinstance(array_a,list):
        array_a = np.array(array_a)
    if isinstance(array_b,list):
        array_b= np.array(array_b)
        
    # Indicates whether the elements of array_a have been found in array_b    
    mask = []
    
    # Compare each element in array_a with the elements in array_b to find matches
    for index in range(array_a.shape[0]):
        comparison = array_a[index,:]
        if in_array(array_b-comparison,rel_tol):
                mask.append(False)
        else:
            mask.append(True)
    
    # Retrieve differences
    differences = array_a[mask]
    
    return differences
        
def line_counter(path):
    '''This function counts the total number of lines in the text files in the folder
    indicated by @path.
    A tuple (total_count,details) is returned. details is a list of tuples with the same
    number of elements as files in the directory specified by @path, where the first 
    entry is the line count and the second entry is the filename'''
    
    def blocks(files, size=65536):
        '''Block read file '''
        while True:
            b = files.read(size)
            if not b: break
            yield b
            
    # List all files in the dir specified by path
    files = glob.glob(path+"/*.txt")
    counts = []
    names = []
    
    for file in files:
        # Keep track of file name
        names.append(file)
        with open(file, "r",encoding="utf-8",errors='ignore') as f:
            # Acumulate counts
            counts.append((sum(bl.count("\n") for bl in blocks(f))))
    # Create a (total_count,details) tuple
    return (sum(counts),list(zip(counts,names)))

def check_sampling_deprecated (sample_indices_set, results, max_score):
    """ Given a set of sample indices, @sample_indices_set, represented as a tuple with
    structure (batch_index, row_index, column_index,scaled_partition_function), this function uses the raw algorithm 
    results to calculate the smallest postive "residual" of the partition function by subtracting
    each score from the scaled_partition_function with coordinates <= (batch_index,row_idx,col_idx). The test checks whether
    incrementing col_idx by 1 results in a negative partition function. If this is the case, then the sampling is correct.
    
    Notes: Raw results are represented as a tuple where the first element is max_score for the particular batch and the second 
    is a matrix containing the eponents of the Gibbs distribution - hence np.sum(np.exp(scores-max_score)) is applied to calculate 
    the scores exactly"""
    
    partition_residuals = []
    
    
    #for sample_indices in sample_indices_set:
    for batch, row_idx, col_idx, partition_function in sample_indices_set:
        
       #  print("Checking scaled partition", partition_function)
        if batch == 0:
            partition_residuals.append( partition_function - np.sum(np.exp(results[batch][1]-max_score).flatten()[0:((row_idx)*(results[batch][1].shape[1]) + col_idx)]) )
            print ("Batch is 0, partition_residual", partition_residuals [-1])
        else:
            for index in range(batch):
                partition_function = partition_function - np.sum(np.exp(results[index][1] - max_score))
         #   print ("After subtracting batch contribution", partition_function) 
            partition_function = partition_function - np.sum( np.exp(results[batch][1] - max_score).flatten()[0:((row_idx)*(results[batch][1].shape[1]) + col_idx + 1)] )
         #   print ("After subtracting rows and columns contribution", partition_function)
            partition_residuals.append(partition_function)  
        
    return partition_residuals

def check_sampling (sample_indices_set, results, max_score):
    """ Given a set of sample indices, @sample_indices_set, represented as a tuple with
    structure (batch_index, row_index, column_index,scaled_partition_function), this function uses the raw algorithm 
    results to calculate the smallest postive "residual" of the partition function by subtracting
    each score from the scaled_partition_function with coordinates <= (batch_index,row_idx,col_idx). The test checks whether
    incrementing col_idx by 1 results in a negative partition function. If this is the case, then the sampling is correct.
    
    Notes: Raw results are represented as a tuple where the first element is max_score for the particular batch and the second 
    is a matrix containing the eponents of the Gibbs distribution - hence np.sum(np.exp(scores-max_score)) is applied to calculate 
    the scores exactly"""
    
    partition_residuals = []
    
    
    #for sample_indices in sample_indices_set:
    for batch, row_idx, col_idx, partition_function in sample_indices_set:
        
       #  print("Checking scaled partition", partition_function)
        if batch == 0:
            partition_residuals.append( partition_function - np.sum(np.exp(results[batch][1] - max_score).flatten()[0:((row_idx)*(results[batch][1].shape[1]) + col_idx + 1)]) )
            print ("Batch is 0, partition_residual", partition_residuals [-1]) 
        else:
            for index in range(batch):
                partition_function = partition_function - np.sum(np.exp(results[index][1]-max_score))
            # print ("After subtracting batch contribution", partition_function) 
            partition_function = partition_function - np.sum( np.exp(results[batch][1] - max_score).flatten()[0:((row_idx)*(results[batch][1].shape[1]) + col_idx + 1)] )
            # print ("After subtracting rows and columns contribution", partition_function)
            partition_residuals.append(partition_function)  
        
    return partition_residuals

def get_synthetic_F_tilde(synthetic_data, dim):
    ''' Computes the contribution of an outcome to the 
    utility function'''
    
    # Compute F_tilde (equation (4.1), Chapter 4, Section 4.1.1)
    # for the synthetic data
    
    const = (1/dim)
    synth_features = synthetic_data[:, :-1]
    synth_targets = synthetic_data[:, -1:].reshape(synthetic_data.shape[0],1)
    F_r = const*synth_features.T@synth_features
    f_r = const*synth_features.T@synth_targets
    F_tilde_r = np.concatenate((F_r,f_r), axis = 1)
    
    return F_tilde_r

def calculate_recovered_scores(synthetic_data_sets, F_tilde_x, scaling_const, dim, max_scaled_utility = 0.0):
    ''' Given a set of @synthetic_data_sets, this function calculates the utilities 
    and scores for the private data set characterised by F_tilde_x. 
    @ max_scaled_utility: Maximum value of scaled utility used to implement exp-normalise trick. Set to 0.0
    by default (exp-normalise not applied)
    @ scaling_constant: Equal to the inverse global sensitivity of the utility times half the 
    privacy parameter epsilon
    @ dim: dimensionality of the private data''' 
    
    # Store F_tilde_r for each synthtic data set
    F_tilde_rs = []
    
    for synthethic_data_set in synthetic_data_sets:
        F_tilde_rs.append(get_synthetic_F_tilde(synthethic_data_set, dim))
        
    F_tilde_rs = np.array(F_tilde_rs)

    # Calculate utitlites
    utilities_array = - np.max(np.abs(F_tilde_x - F_tilde_rs), axis = (2,1))
    
    scaled_utilities_array = - scaling_const*np.max(np.abs(F_tilde_x - F_tilde_rs), axis = (2,1))
    
    # Calculate scores
    scores_array = np.exp(scaled_utilities_array - max_scaled_utility)
    
    return (scores_array, scaled_utilities_array, utilities_array)

def retrieve_scores_from_results(results, sample_indices, max_scaled_utility = 0.0):
    ''' This test function is used to calculate the scores given 
    @results, a data structure containing tuples formed of scaled
    utility matrices for each batch [element at index 1] and the max.
    scaled utility for that batch [element at index 0]. If max_scaled_utility is
    set to 0.0, exp-normalise trick is not applied.'''
    
    scores = []
    
    batch_idxs = [element[0] for element in sample_indices]
    row_idxs = [element[1] for element in sample_indices]
    col_idxs = [element[2] for element in sample_indices]
    
    # Remember that max_score has not been subtracted from each result and 
    # that exp was not taken
    score_results = [(element[0], np.exp(element[1] - max_scaled_utility)) for element in results ]
    
    # Look-up the scores in the results data structures
    for batch_idx, row_idx, col_idx in zip(batch_idxs, row_idxs, col_idxs):
        scores.append(score_results[batch_idx][1][row_idx,col_idx])
    
    return scores

def load_batch_scores(path):
    ''' Returns the contents of the file specified by absolute path. 
    
    This is the baseline version used to test the netmechanism module.'''
    with open(path, "rb") as data:
        batch_scores = pickle.load(data)
    return batch_scores

def retrieve_scores(filenames, batches = []):
    """ This function unpickles the files in specified in the 
    filenames list, returning a list containing the contents of the unpickled files.
    If if batches list is specified, then only the files to the corresponding
    to entries of the list are loaded 
    
    This is the baseline version used to test the netmechanism module."""
    
    def get_batch_id(filename):
        return int(filename[filename.rfind("_") + 1:])
         
    data = []
    
    # Filenames have to be sorted to ensure correct batch is extracted
    filenames  = sorted(filenames, key = get_batch_id)
    
    if not batches:        
        for filename in filenames:
            data.append(load_batch_scores(filename))
    else:
        for entry in batches:
            data.append(load_batch_scores(filenames[entry]))
    return data

def load_data(path):
    ''' Returns the contents of the file specified by absolute path '''
    with open(path,"rb") as container:
        data = pickle.load(container)
    return data

def compute_second_moment_utility(outcomes, targets, dim, F_tilde_x, scaling_const):
                         
    f_r_tensor = (1/dim)*np.matmul(targets, outcomes)
    
    # Calculate F_r = 1/d Xh'Xh (' denotes transpose). This is applied for all Xh in the synth_features_tensor
    F_r_tensor = (1/dim)*np.transpose(outcomes,axes = (0,2,1))@outcomes
    
    #TODO: add comment
    f_r_expand = f_r_tensor.reshape(tuple([*f_r_tensor.shape,1]))
    
    #TODO: add comment
    F_r_expand = np.repeat(F_r_tensor, repeats = targets.shape[0], axis = 0).reshape(F_r_tensor.shape[0], -1, *F_r_tensor[0].shape)
    
    #TODO: add comment
    F_tilde_r = np.concatenate((F_r_expand, f_r_expand), axis = 3)
    
    # Utilities for the particular batch are returned as a matrix of dimension batch_size x p where p is the number of 
    # synthetic targets. Exp-normalise trick is implemented so the exponentiation is done in the sampling step
    utility = - scaling_const*np.max(np.abs(F_tilde_x - F_tilde_r), axis = (3,2))
    
    return utility

def get_private_F_tilde (private_data):
    
    # Compute F_tilde (equation (4.1), Chapter 4, Section 4.1.1)
    # for the private data
    
    const = (1/private_data.features.shape[0])
    F_x = const*private_data.features.T@private_data.features
    f_x = const*private_data.features.T@private_data.targets
    F_tilde_x = np.concatenate((F_x,f_x), axis = 1)
    
    return F_tilde_x  

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

def evaluate_sample_score(batch_index, features, targets, scaling_const, F_tilde_x, dim, batch_size, base_filename_s,\
                          directory, test = False):
    
    # Storage structure
    struct = {}
    
    # Store the batch index to be able to retrieve the correct sample during sampling step
    struct['batch_index']  = batch_index
    
    # Generate a batch of combinations according to the batch_index
    batch = list(itertools.islice(itertools.combinations(range(features.shape[0]),dim),(batch_index)*batch_size,(batch_index+1)*batch_size))
    
    # Evaluate utility - note that exponential is not taken as sum-exp trick is implemented to 
    # evalute the scores in a numerically stable way during sampling stage
    score_batch = compute_second_moment_utility(features[batch,:], targets, dim, F_tilde_x, scaling_const)
    struct ['scores'] = score_batch
    struct ['test_data'] = batch
    
    # Create data structure which stores the scores for each batch along with 
    # the combinations that generated them
    max_util = np.max(score_batch)
    
    # Find the indices of the maximum in the score batch
    max_scaled_util_ind = np.argwhere( np.isclose(score_batch - max_util, 0.0, rtol = 1e-9)).tolist()
    
    # Insert the index of the batch to have the coordinates of the sample with max utility
    for elem in max_scaled_util_ind:
        elem.insert(0, batch_index)
    max_scaled_util_coord = [tuple(elem) for elem in max_scaled_util_ind]
    
    
    # save the slice object
    filename = "/" + base_filename_s + "_" + str(batch_index)
    save_batch_scores(struct,filename,directory)
    
    partial_sum = np.sum(np.exp(score_batch))
    # Only max_util is returned in the final version of the code to 
    # allow implementation of exp-normalise trick during sampling . 
    # score_batch is returned for testing purposes
    
    return (max_util,score_batch, partial_sum, max_scaled_util_coord)

def recover_synthetic_datasets(sample_indices, features, targets, batch_size, dim):
    
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
        recovered_slice = itertools.islice(itertools.combinations(range(features.shape[0]), dim), \
                                           (batch_idx)*batch_size, (batch_idx + 1)*batch_size)
        
        # Recover the correct combination 
        combination = nth(recovered_slice, comb_idx)
        # print ("Recovered combination", combination)
    
        # Recover the feature matrix
        feature_matrices.append(features[combination, :])

    # Reconstruct the targets for the synthethic feature matrix 
    for feature_matrix,target_index in zip(feature_matrices,target_indices):
        #try:
        synthetic_data_sets.append(np.concatenate((feature_matrix, targets[target_index,:].reshape(targets.shape[1], 1)), axis = 1))
        # except IndexError:
        #    synthetic_data_sets.append(np.concatenate((feature_matrix, targets[target_index - 1,:].reshape(targets.shape[1],1)), axis = 1))
        
    return synthetic_data_sets 

def get_optimal_datasets(results, features, targets, batch_size, dim):
    # First we extract the maximum scaled utilities for each batch in an array
    partial_maxima = np.array([elem[0] for elem in results])
    # Then we return the indices in the array where the maxima occur - the maximum might exist in multiple batches
    maxima_indices = np.argwhere( np.isclose(partial_maxima - np.max(partial_maxima), 0.0, rtol = 1e-9))
    maxima_indices = list(chain.from_iterable(maxima_indices))
    # Now we just merge the tuples of indices where the maximum scaled utilities have been identified in combs_array
    combs_array = []
    for index in maxima_indices:
        combs_array.extend(results[index][3])
    # And finally we reconstruct the datasets which generate the corresponding scaled utilities
    synthetic_datasets = recover_synthetic_datasets(combs_array, features, targets, batch_size, dim)
    return synthetic_datasets
