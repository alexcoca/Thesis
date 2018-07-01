# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 18:29:47 2018

@author: alexc
"""

def construct_batches_deprecated(n,k,batch_size):
    '''Function works correctly but two copies of each slice are needed.
    This function does not work correctly. If one sets n=10, k=4 and then,
    creates a list out of e.g. combinations_slices[4], then this actually returns
    the first batch_sizes elements from the iterator (weirdly)'''
    
    
    combinations_slices = []
    
    # Calculate number of batches
    n_batches = math.ceil(comb(n,k,exact=True)/batch_size)
    
    # Construct iterator for combinations
    combinations = itertools.combinations(range(n),k)
    
    # Slice the iterator into n_baches slices
    while len(combinations_slices) < n_batches:
        combinations_slices.append(itertools.islice(combinations,batch_size))
    
    return combinations_slices

def construct_batches(n,k,batch_size):
    ''' Function does not work correctly - see doc for construct_batches_deprecated
    to understand why'''
    
    combinations_slices = []
    
    # Calculate number of batches
    n_batches = math.ceil(comb(n,k,exact=True)/batch_size)
    
    # Construct iterator for combinations
    combinations = itertools.combinations(range(n),k)
    
    # Slice the iterator into n_batches slices. Each slice is duplicated
    # so that it can be subsequently written to a file for later retrieval 
    # during sampling. This is necessary since calling the utility calculation
    # routine exahausts the original slice. If the iterator is first converted to list
    # then storage requirement increases (e.g. from 1KB to 800+ kB for a list of
    # 50,000 tuples of dimension 4)
    
    while len(combinations_slices) < n_batches:
        combinations_slices.append(itertools.tee(itertools.islice(combinations,batch_size)))
        
    return combinations_slices

def evaluate_sample_score(batch_index):
    '''Original code. Investigation revealed that the batches list does not behave correctly,
    retrieving the batch with batch_index is likely not going to return the combinations between
    (batch_index - 1)*batch_size and (batch_index)*batch_size''' 
    
    
    # Storage structure
    struct = {}
    
    # Store combinations to be able to retrieve the correct sample g
    struct['combs']  = batches[batch_index][0]
    
    # Evaluate utility - note that exponential is not taken as sum-exp trick is implemented to 
    # evalute the scores in a numerically stable way during sampling stage
    score_batch = scaling_const*compute_second_moment_utility(features[list(batches[batch_index][1]),:])
    struct ['scores'] = score_batch
    
    # Create data structure which stores the scores for each batch along with 
    # the combinations that generated them
    max_util = np.max(score_batch)
    
    # save the slice object
    filename = "/"+base_filename_s + "_" + str(batch_index)
    save_batch_scores(struct,filename,directory)
    
    # Only max_util is returned in the final version of the code to 
    # allow implementation of exp-normalise trick during sampling . 
    # score_batch is returned for testing purposes
    
    return (max_util,score_batch)


# The test below is no longer necessary as the combinations are not saved any longer
  
# Now let's look at whether the tee reloading works. Well take the example of batch zero

print("Retrieved " + str(len(list(reloaded_data_batches[0]['combs']))) + " combinations for batch 0")

# Test that there is no empty list that we reloaded

combs_lengths = []

for element in reloaded_data:
    combs_lengths.append(len(list(element['combs'])))

# Assert no empty lists
    
assert (0 not in [1,2])

counter = collections.Counter(combs_lengths)

# Calculate outcome space size to ensure correctness
num_targets = targets.shape[0]

outcome_space_size_est = sum([num_targets*x*y for x,y in zip(list(counter.values()),list(counter.keys()))])

outcome_space_size_calc = est_outcome_space_size(features.shape[0],dim,num_points_targ)

assert outcome_space_size_calc == outcome_space_size_est