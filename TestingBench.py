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

#dim = 3
#num_points = 10
#upper_bound = 1.0
#lower_bound = -1.0
#radius = 1.0
#num_dec = 10
#r_tol = 1e-5
#OutputLattice = FeaturesLattice()
#OutputLattice.generate_l2_lattice(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points,pos_ord=True,rel_tol=r_tol)
#intersection_m2 = testutilities.bruteNonIntegerIntersection(dim=dim,radius=radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound,filtered = False,r_tol=r_tol)
#test_points = OutputLattice.points
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
from netmechanism import FeaturesLattice
import pickle

dim = 3
num_points = 12
upper_bound = 1.0
lower_bound = -1.0
radius = 1.0
num_dec = 10
folder_path = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Lattices'
lattice_name='d_3_npt_12'
r_tol = 1e-5
OutputLattice = FeaturesLattice()
OutputLattice.generate_l2_lattice(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points,pos_ord=True,rel_tol=r_tol)
original_data = OutputLattice.points
OutputLattice.save_lattice(folder_path=folder_path,lattice_name=lattice_name)
basepath = folder_path+"/"+lattice_name
with open(basepath,"rb") as data:
    reloaded_data = pickle.load(data)
