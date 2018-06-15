# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:51:56 2018

@author: alexc
"""
import math
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import itertools


class L2Lattice():
    
    def __init__(self):
        self.dim = 2 
        self.radius = 1
        self.points = []
    
    def ordered_recursion(self,pos_lattice_coord, radius, dim,upper_bound):
        ''' This function recursively determines all ordered solutions of dimension d
        (x_1,x_2,...,x_d) where s.t. x_i <= upper_bound for all i in [d]. The ordering chosen is x_1>=x_2>=...>=x_d)'''
        partial_solutions = []
        for coordinate in reversed([entry for entry in pos_lattice_coord if entry <=upper_bound]):
            if (dim == 1):
                partial_solutions.append([coordinate])
            else:
                for point in self.ordered_recursion(pos_lattice_coord,radius,dim-1,upper_bound): 
                    # Ensures the ordering is satisfied
                    if coordinate <= point[-1]: 
                        # TODO: we should not check if this is a solution technically 
                        # So final code should not contain this
                        # Ensure this is a valid solution
                        if sum([i**2 for i in [coordinate]+point]) <= radius: 
                            partial_solutions.append(point+[coordinate])
        # TODO: Test & remove assertion 
        if not partial_solutions:
            assert False
        return partial_solutions
    
    def main_recursion(self,radius,dim,pos_lattice_coord,lb,ub,x_prev,max_dim):
        points = []
#        if dim == max_dim:
#            radius = 1.0
#            ub = radius
#            lb = math.sqrt(radius**2/dim)
#            x_prev= []
        current_x_range =  [entry for entry in pos_lattice_coord if entry > lb and entry <=ub]
        for x in reversed(current_x_range):
            # Update radius
            if radius**2 < x**2:
                continue
            else:
                radius = math.sqrt(radius**2- x**2)
                lb = math.sqrt(radius**2/(dim-1))
                ub = x
                x_prev.append(x)
            if dim == 2: 
                # maybe we should think of dim == 2 as a base case? 
                # Would dim == 1 make sense? Is the correct condition
                # indeed min(x,lb)?
                for entry in pos_lattice_coord:
                    if entry <= min(x,lb):
                        points.append([entry])
            else:
                for partial_entry in self.main_recursion(radius,dim-1,pos_lattice_coord,lb,ub,x_prev,max_dim):
                    candidate = x_prev+partial_entry
                    if sum([i**2 for i in candidate]) <= 1.0:
                        self.points.append(candidate)    
                low_d_soln = self.ordered_recursion(pos_lattice_coord,radius,dim-1,lb)
                if low_d_soln:
                    for partial_soln in low_d_soln:
                        candidate = x_prev[0:max_dim-(dim-1)]+partial_soln
                        assert sum([i**2 for i in candidate]) <= 1.0
                        self.points.append(candidate)
                radius = 1.0 
                x_prev = []
                lb=math.sqrt(radius/dim)
                ub = radius
        return points
    
    def generate_permuted_solutions(self,points):
        solutions = []
        for point in points:
                solutions.extend(list(multiset_permutations(point)))
        return solutions
    
    def generate_signed_solutions(self,points,dim):
        
        def generate_signs(dim):
            signs = []
            for value in range(2**dim):
                signs.append([int(x) for x in list(bin(value)[2:].zfill(dim))])
            return signs
        
        signs_list = generate_signs(dim)
        solutions = []
        for point in points:
            # Signed solutions handled separately if there are coordinates = 0.0
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
                # The signs combination is represented as a binary number where 0 is - and 1 is + 
                for sign_combination in signs_list:
                    solutions.append([-x if y == 0 else x*y for x,y in zip(point,sign_combination)])
        return solutions    
        
    
    def generate_l2_lattice(self,dim=2,radius=1,lower_bound=-1,upper_bound=1,num_points=5):
        
        # Find lattice coordinates
        full_lattice_coord = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True)
        # Extract positive coordinates
        pos_lattice_coord = list(full_lattice_coord[full_lattice_coord >= 0.0])
        self.points.extend(self.main_recursion(radius,dim,pos_lattice_coord,lb=math.sqrt(radius/dim),ub=radius,x_prev = [],max_dim=dim))
        self.points.extend(self.ordered_recursion(pos_lattice_coord,radius,dim,math.sqrt(radius/dim)))
        self.points = self.generate_permuted_solutions(self.points)
        self.points = self.generate_signed_solutions(self.points,dim)
        


