# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:00:44 2018

@author: alexc
"""
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math 
import itertools
from sympy.utilities.iterables import multiset_permutations

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
    def recurse_solutions(pos_lattice_coord, radius, dim,upper_bound):
        ''' This function recursively determines all the solutions with the property 
        x1>= x2>=...>=xd s.t. x_i <= upper_bound) '''
        
        partial_solutions = []
        for coordinate in reversed([entry for entry in pos_lattice_coord if entry <=upper_bound]):
            if (dim == 1):
                partial_solutions.append([coordinate])
            else:
                for point in recurse_solutions(pos_lattice_coord,radius,dim-1,upper_bound): 
                    if coordinate <= point[-1]: # Ensures the ordering is satisfied
                        if sum([i**2 for i in [coordinate]+point]) <= radius^2: # Ensure this is a valid solution
                            partial_solutions.append(point+[coordinate])
        if not partial_solutions:
            assert False
        return partial_solutions

    def main_recursion(x_range,radius,dim,bound,pos_lattice_coordinates):
        points = []
        for x in x_range:
            # bound = math.sqrt(radius**2/dim)
            # Update radius
            if dim*bound**2 >= x**2: # This is a bit hacky?
                if dim > 1:
                    bound = math.sqrt((dim*bound**2-x**2)/(dim-1)) # Should the radius not be reset to its starting point when we 
                # Generate lower dimensional solutions with (*) property
                    x_range = [coordinate for coordinate in pos_lattice_coordinates if coordinate <= x and coordinate > bound]
            # Base case: # TODO: correct this, probably not right
            if dim == 1: # 
                possible_solution_lower = [coordinate for coordinate in pos_lattice_coordinates if coordinate <= x and coordinate > bound ]
                if possible_solution_lower:
                    points.append(possible_solution_lower)
                # possible_solution_upper = recurse_solutions(pos_lattice_coord,radius,dim,bound)
                # if possible_solution_upper:
                #    points.append(possible_solution_upper)
            else:
                for partial_entry in main_recursion(x_range,radius,dim-1,bound,pos_lattice_coordinates):
                    # Track solutions for which x_i < ub
                    low_d_soln = recurse_solutions(pos_lattice_coord,radius,dim-1,bound)
                    if low_d_soln:
                        for partial_solution in low_d_soln:
                            candidate = [x]+partial_solution
                            if sum([i**2 for i in candidate]) <= radius**2:
                                points.append(candidate)
                    # Track the solutions for which x_i > ub (aka for which x is in x_range)
                    candidate = [x]+partial_entry
                    if sum([i**2 for i in candidate]) <= radius**2:
                        points.append(candidate)                    
        return points
    
    def generate_signed_solutions(points,dim):
        
        def generate_signs(dim):
            signs = []
            for value in range(2**dim):
                signs.append([int(x) for x in list(bin(value)[2:].zfill(dim))])
            return signs
        
        def remove_duplicates(k):
          newk = []
          for i in k:
            if i not in newk:
              newk.append(i)
          return newk        
                
        signs_list = generate_signs(dim)
        solutions = []
        for point in points:
            if 0.0 in point:
                temp_sign_list = []
                temp = []
                # Find 0 indices 
                zero_indices = [i for i,e in enumerate(point) if e == 0.0]
                temp_sign_list = generate_signs(dim-len(zero_indices))
                for sign_combination in temp_sign_list:
                    temp.append([-x if y == 0 else x*y for x,y in zip([entry for entry in point if entry != 0.0],sign_combination)])
                for zero_free_soln in temp:
                    for index in zero_indices:
                        zero_free_soln.insert(index,0.0)
                    solutions.append(zero_free_soln)
            else:
                for sign_combination in signs_list:
                    solutions.append([-x if y == 0 else x*y for x,y in zip(point,sign_combination)])
        return solutions
            
    def generate_permuted_solutions(points):
        solutions = []
        for point in points:
                solutions.extend(list(multiset_permutations(point)))
        return solutions
                
    ub = math.sqrt(radius/dim)
    # Determine all solutions with x1>=x2>=...>=xd and x_i <= ub for all i \in [d] (*)
    points.extend(recurse_solutions(pos_lattice_coord,radius,dim,ub))
    # Determine range for x1
    x_range = [i for i in pos_lattice_coord if i > ub]
    # Recursively determine the remainder of the solutions
    points.extend(main_recursion(x_range,radius,dim,ub,pos_lattice_coord))
    points = generate_permuted_solutions(points)
    points = generate_signed_solutions(points,dim)
    
    
    return points

def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1):
    coord_array = list(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True))
    lattice_coordinates = [coord_array for i in range(dim)]
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    return (intersection,coord_array) 

dim = 2
num_points = 5
upper_bound = 1
lower_bound = -1
radius = 1.0
points = generateIntersection(dim=2,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points)
intersection_m2,coord_array_m2 = bruteNonIntegerIntersection(dim=dim,radius=radius,num_points=num_points,lower_bound=lower_bound,upper_bound=upper_bound)
#%% 

