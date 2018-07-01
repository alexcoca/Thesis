# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:12:17 2018

@author: alexc
"""
from data_generators import ContinuousGenerator
import numpy as np 
from sklearn.preprocessing import normalize
import pdb

# Fix random number generator
# np.random.seed(25)

# Exploration into the sensitivity calculation...

# Define some data generation parameters
n_feat = 20
dimensionality = 4
var= 1
seed = 24
target_bound = 1
# Generate a continuous data set... with row 2-norm bounded by 1. This seams to
# force all elements to be positive

DataGenerator = ContinuousGenerator(d=dimensionality,n=n_feat,variance=var)
DataGenerator.generate_data(seed = seed,bound_recs=True)
features = DataGenerator.features
targets = DataGenerator.targets
assert np.all(np.abs(targets) <= target_bound)

# Create features object where the normalisation is not applied (features are both
# positive and negative)

DataGenerator_2 = ContinuousGenerator(d=dimensionality,n=n_feat,variance=var)
DataGenerator_2.generate_data(seed=seed,bound_recs=False)
features_2 = DataGenerator_2.features
targets_2 = DataGenerator_2.targets
assert np.all(np.abs(targets_2) <= target_bound)
# Select a row at random
pdb.set_trace()
row_idx = np.random.choice(range(n_feat))

# The question to answer is how much can the infinity norm of the x_n^T*x_n grow...

record = features[None,row_idx,:]
record_target = targets[row_idx]

# Calculate outer prod max for original vector
l_inf_rec = np.max(np.abs(record@record.T))
print("For the orig. record, outer prod is",l_inf_rec)
print("The norm of the original record is",np.linalg.norm(record,ord=2))

# Normalise record
record_norm = normalize(record)

# Calculate outer prod max for norm vector
l_inf_rec_norm = np.max(np.abs(record_norm@record_norm.T))
print("For the norm. record, outer prod is",l_inf_rec_norm)

# Let's just add a minor perturbation to the original record to see what happens with the outer product

# Postive perturbation
sign = 1 # perturbation sign
pert = 0.1 # perturbation magnitude
pert = pert*sign 
index = np.random.choice(dimensionality) # choose an index to perturb
record[0,index] = record[0,index]+ pert
print("Norm of perturbed record is", np.linalg.norm(record,ord=2))
 
# Calculate the outer product for the perturbed fector 
l_inf_rec_p = np.max(np.abs(record@record.T))
print("For the perturbed record, outer prod is",l_inf_rec_p)
print("For the perturbed record, the norm is",np.linalg.norm(record,ord=2))

# Generate a random vector with elements close to zero and normalise it

rand_vec = np.random.normal(scale=1e-40,size=(1,4))
rand_vec_norm = normalize(rand_vec)
print("Random vector",rand_vec)
print("Random vector normalised",rand_vec_norm)
print("Outer product orig vec",rand_vec.T@rand_vec)
print("Outer product norm vec",rand_vec_norm.T@rand_vec_norm)
print("Outer product difference",rand_vec_norm.T@rand_vec_norm-rand_vec.T@rand_vec)
print("Infinity norm of the outer product difference",np.max(np.abs(rand_vec_norm.T@rand_vec_norm-rand_vec.T@rand_vec)))