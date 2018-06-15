# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:19:45 2018

@author: alexc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:01:08 2018

@author: alexc
"""
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math 
import itertools
from sympy.utilities.iterables import multiset_permutations

def generateIntersection(dim=2,radius=1,lower_bound=-1,upper_bound=1,num_points=5):
    points = []
    # Generate lattice coordinates
    full_lattice_coord = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True)
    
    # Extract positive coordinates
    pos_lattice_coord = list(full_lattice_coord[full_lattice_coord >= 0.0])
    
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
                        if sum([i**2 for i in [coordinate]+point]) <= radius**2: # Ensure this is a valid solution
                            partial_solutions.append(point+[coordinate])
        if not partial_solutions:
            assert False
        return partial_solutions

    def main_recursion(radius,dim,pos_lattice_coord,lb,ub,x_prev): 
        points = []
        # current_x_range: This becomes each variable in turn.
        # First we fix x_1 to be in the interval (sqrt(r^2/d),r]. We take the largest of the 
        # x_1s and perform a trick to get partial soln (x2,x3,...,xd) with x_i < sqrt((r^2-x_1^2)/(d-1))/
        # We append the selected x_1 to each of these solutions and obtain a set of solutions for the problem.
        # We then select the largest value of x_2 that is <=x1 but larger than sqrt((r^2-x_1^2)/(d-1))  and obtain (x3,x4,...xd) with x_i < sqrt((r^2-x_1^2-x_2^2)/(d-2))
        # The d-2 solutions should be appended to the vector [x1,x2]. 
        
        # Eventually, we will get to a case where d=2. Hence we only have x_d left so we select all the x_d values
        # that are < = min(x_{d-1},lb). lb is what we updated throughout the whole recursive stack, it should be \sqrt((r_final^2)/1)
        # where r_final= sqrt(r_orig(=1)-x1^2-x2^2-....-x_{d-1}^2). But we kept the x_1,x_2,...,x_{d-1} we fixed in a vector 
        # so when we append them to the values of x_d to this vector to obtain our solutions.
        
        # At this stage, we have to take the next value of x_1. But we won't be able to get correct results unless we 
        # get back to the values of the parameters for r,lb,ub that we had when we started the recursion ...
        
        # The recursion continues until ???
        
        
        current_x_range =  [entry for entry in pos_lattice_coord if entry > lb and entry <=ub]
        for x in reversed(current_x_range):
            # bound = math.sqrt(radius**2/dim)
            # Update radius
            if radius**2 < x**2:
                continue
            else:
                radius = math.sqrt(radius**2- x**2)
                lb = math.sqrt(radius**2/(dim-1))
                ub = x
                x_prev.append(x)
            if dim == 2: # maybe we should think of dim == 2 as a base case?
                for entry in pos_lattice_coord:
                    if entry <= min(x,lb):
                        points.append([entry])
            else:
                for partial_entry in main_recursion(radius,dim-1,pos_lattice_coord,lb,ub,x_prev):
                    # lb = math.sqrt(radius**2/(dim-1))
                    # ub = x
                    # x_prev.append(x) # or extend?
                    # Track the solutions for which x_i > ub (aka for which x is in x_range)
                    candidate = [x]+partial_entry
                    if sum([i**2 for i in candidate]) <= radius**2:
                        points.append(candidate)    
                low_d_soln = recurse_solutions(pos_lattice_coord,radius,dim-1,lb)
                if low_d_soln:
                    for partial_soln in low_d_soln:
                        candidate = x_prev+partial_soln
                        assert sum([i**2 for i in candidate]) <= 1.0
                        points.append(candidate)
                radius = 1.0 
                x_prev = []
                lb=math.sqrt(radius/dim)
                ub = radius
                
                
                        
        return points
    
    
    # Determine all solutions with x1>=x2>=...>=xd and x_i <= ub for all i \in [d] (*)
    # points.extend(recurse_solutions(pos_lattice_coord,radius,dim,ub))
    # Determine range for x1
    #x_range = [i for i in pos_lattice_coord if i > ub]
    # Recursively determine the remainder of the solutions
    points.extend(main_recursion(radius,dim,pos_lattice_coord,lb=math.sqrt(radius/dim),ub=radius,x_prev = []))
    
    return points

dim = 3
num_points = 7
upper_bound = 1
lower_bound = -1
radius = 1
points = generateIntersection(dim=dim,radius=radius,lower_bound=lower_bound,upper_bound=upper_bound,num_points=num_points)