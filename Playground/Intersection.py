# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:00:44 2018

@author: alexc
"""
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math 
#%%
# Very simple implementation? Is it even correct?
def generateIntegerIntersection(dim=2,radius=6):
    points = []
    for coord in range(int(radius)+1):
        newr = np.sqrt(radius**2-coord**2)
        if dim == 1:
            points.append([coord])
        else:
            for point in generateIntegerIntersection(dim-1,radius):
                candidate = [coord]+point
                if np.linalg.norm(candidate,ord=2) <=radius:
                    points.append(candidate)
    return points


def generateIntegerIntersection(dim=2,radius=6):
    points = []
    for coord in range(int(radius)+1):
        newr = np.sqrt(radius**2-coord**2)
        if dim == 1:
            points.append([coord])
        else:
            for point in generateIntegerIntersection(dim-1,newr):
                    points.append([coord]+point)
    return points

def bruteIntegerIntersection(dim=2,radius=5):
    coord_array = [range(radius+1) for i in range(dim)]
    flat_grid = np.array(np.meshgrid(*coord_array)).T.reshape(-1,dim)
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    return (intersection,flat_grid) 

radius = 5
dim = 2
intersection_m1 = np.array(generateIntegerIntersection(dim=dim,radius=radius))
# Test if there is a norm with bad value
assert np.any(np.linalg.norm(intersection_m1,ord=2,axis=1) < radius)

# Brute force intersection
intersection_m2,flat_grid = bruteIntegerIntersection(dim=dim,radius=radius)

assert np.any(np.linalg.norm(intersection_m2,ord=2,axis=1) < radius)

# Test the two methods yield the same result
assert np.any(intersection_m1 == intersection_m2)
#%% Which look-up is faster? As a hash table or as a simple numpy array?
def array_lookup(array,indices):
    look_up=array[indices,:]
    return look_up

def dict_lookup(dictionary,indices):
    look_up = itemgetter(*indices)(dictionary)
    look_up = np.array(look_up)
    return look_up

def dict_lookup2(dictionary,indices):
    look_up = [dictionary[x] for x in indices]
    look_up = np.array(look_up)
    return look_up

list_size = 8
max_dim = 3834490
indices = np.random.choice(range(max_dim),list_size)

# Conclusion: 
# array_lookup: 1.73 µs ± 8.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
# dict_lookup: 7.34 µs ± 139 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# dict_lookup (without conversion to np array): 2.34 µs ± 16.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
# dict_lookup2: 7.59 µs ± 137 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)


#%%
# Now let's extend to the rational case 
def generateNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1):
    points = []
    coord_array = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True)
    for coord in coord_array :
        if dim == 1:
            points.append([coord])
        else:
            for point in generateNonIntegerIntersection(dim-1,radius):
                # candidate = [coord]+point
                # if np.linalg.norm(candidate,ord=2) <=radius:
                    points.append([coord]+point)
    return (points,coord_array)

def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1):
    coord_array = list(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True))
    lattice_coordinates = [coord_array for i in range(dim)]
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    return (intersection,coord_array) 


radius = 1
dim = 2
num_points = 10
upper_bound = 1
lower_bound = 0

intersection_m1,coord_array_m1 = np.array(generateNonIntegerIntersection(dim,radius,num_points = num_points, upper_bound=upper_bound,lower_bound=lower_bound))
intersection_m2,coord_array_m2 = bruteNonIntegerIntersection(dim,radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound)


#%% 

def generateIntersection(dim=2,radius=1,lower_bound=-1,upper_bound=1,num_points=5):
    points = []
    # Generate lattice coordinates
    full_lattice_coord = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True)
    
    # Extract positive coordinates
    pos_lattice_coord = list(full_lattice_coord[full_lattice_coord >= 0.0])
    
    
    # Define resurse_solutions, recursive procedure to compute solutions with x1>= x2>=...>=xd s.t. x_i <= sqrt(radius^2/dim)
    def recurse_solutions(pos_lattice_coord,radius,dim,lower_bound,upper_bound):
        partial_solutions = []
        for coordinate in reversed([entry for entry in pos_lattice_coord if entry <=bound]):
            if (dim == 1):
                partial_solutions.append([coordinate])
            else:
                for point in recurse_solutions(pos_lattice_coord,radius,dim-1,bound): 
                    if coordinate <= point[-1]:
                        if sum([i**2 for i in [coordinate]+point]) <= radius^2:
                            partial_solutions.append(point+[coordinate])
        return (partial_solutions,lb)
                            
    ub = math.sqrt(radius/dim)
    points.append(recurse_solutions(pos_lattice_coord,radius,dim,ub))
    # Determine solutions for which x_1 is greater than the upper bound 
    for coord_value in [i for i in partial_solutions if i > ub]:
        radius = math.sqrt(radius**2-coord_var**2)/(dim-1))
        recurse_solutions(pos_lattice_coord,radius,dim-1,ra)
    return points

generateIntersection(dim=2,radius=1,lower_bound=-1,upper_bound=1,num_points=9)

