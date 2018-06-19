# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:51:56 2018

@author: alexc
"""
import math
import numpy as np
from sympy.utilities.iterables import multiset_permutations

class FeaturesLattice():
    
    def __init__(self):
        self.dim = 2 
        self.radius = 1
        self.points = []

    
    def ordered_recursion(self,pos_lattice_coord, radius, dim,upper_bound):
        ''' This function recursively determines all ordered solutions of dimension d
        (x_1,x_2,...,x_d) where s.t. x_i <= upper_bound for all i in [d]. The ordering chosen is x_1>=x_2>=...>=x_d)'''
        partial_solutions = []
        for coordinate in reversed([entry for entry in pos_lattice_coord if entry <=upper_bound]):
            if dim == 1:
                partial_solutions.append([coordinate])
            else:
                for point in self.ordered_recursion(pos_lattice_coord,radius,dim-1,upper_bound): 
                    
                    # Ensures the ordering is satisfied
                    if coordinate <= point[-1]:# or math.isclose(coordinate,point[-1],rel_tol=rel_tol): 
                        candidate = point+[coordinate]
                        candidate_norm = np.linalg.norm(candidate,ord=2)
                        # Techincally, this if could be excluded? TODO: Exclude and re-test code.
                        if math.isclose(candidate_norm,radius,rel_tol=self.rel_tol) or candidate_norm <= radius :
                            partial_solutions.append(point+[coordinate])
        # NB It is possible that solutions do not exist: e.g. if x_1 = 1 and we have even number of points
        #  , then there will not be any solution!                    
        return partial_solutions
    
    def main_recursion(self,radius,dim,pos_lattice_coord,lb,ub,x_prev,max_dim):
        
        # Determine the range of the variable currently being recursed on. If dim == max_dim
        # then x is x_1. Otherwise x is x_2 (if dim==max_dim-1) and so on.... These are the points inside
        # a square inscribed in the top-right quadrant
        current_x_range =  [entry for entry in pos_lattice_coord if entry > lb and entry <=ub]
        
        for x in reversed(current_x_range):
            # Update radius: this update accounts for the fact that fixing x limits the range
            # of the remaining coordinates
            if radius**2 < x**2: 
                # For numerical stability, the cases where the points are close to the sphere are handled separately
                if math.isclose(radius**2,x**2,rel_tol=self.rel_tol):
                    radius = 0.0
                    lb = 0.0
                    ub = x
                    x_prev.append(x)
                    
            # This means the particular combination is not a valid solution so we skip it. The recursion will instead
            # try to find solutions by setting the remainder of the coordinates to be <= \sqrt(r^/d), where r,d are the 
            # current values of the radius and the current dimension
                else:
                    continue
            else:
                radius = math.sqrt(radius**2- x**2)
                lb = math.sqrt(radius**2/(dim-1))
                ub = x
                x_prev.append(x)
            if dim == 1: 
                if len(x_prev) == max_dim:
                    assert math.isclose(np.linalg.norm(x_prev,ord=2),1.0,rel_tol=self.rel_tol)
                    self.points.append(x_prev)
            else:
                # Recursive call to solve a lower dimensional problem, with updated radius and dimension
                self.main_recursion(radius,dim-1,pos_lattice_coord,lb,ub,x_prev,max_dim)
                
                # Recover solutions in dim-1 that satisfy Procedure 1 (see, Algorithm TBD, report)
                low_d_soln = self.ordered_recursion(pos_lattice_coord,radius,dim-1,lb)
                
                # Search for lower dimensional solutions given the coordinates fixed so far
                if low_d_soln:
                    for partial_soln in low_d_soln:
                        candidate = x_prev[0:max_dim-(dim-1)]+partial_soln
                        assert math.isclose(np.linalg.norm(candidate,ord=2),1.0,rel_tol=self.rel_tol) or (np.linalg.norm(candidate,ord=2) <= 1.0)
                        self.points.append(candidate)
                        
                # Update the radius and bounds after performing the computations for a particular dim.
                # so that they have the correct values when computing solutions up the recursion stack,
                # in higher dimensions
                radius = math.sqrt(radius**2+x**2)
                x_prev = x_prev[:(max_dim-dim)]
                lb=math.sqrt(radius/dim)
                ub = radius
    
    def generate_permuted_solutions(self,points):
        """"Generates all the permutations of the solutions contained in the 
        array @points"""
        solutions = []
        
        # Since the solutions can contain identical elements, used multiset permutations
        # to avoid duplicating solutions
        for point in points:
                solutions.extend(list(multiset_permutations(point)))
        return solutions
    
    def generate_signed_solutions(self,points,dim):
        """ Generates all the signed solutions given the positive solutions in the array
        @points"""
        def generate_signs(dim):
            signs = []
            for value in range(2**dim):
                signs.append([int(x) for x in list(bin(value)[2:].zfill(dim))])
            return signs
        
        # Solutions container
        solutions = []
        
        # Generate all possible signs combinations for the solutuions 
        signs_list = generate_signs(dim)
    
        for point in points:
            # Signed solutions handled separately if there are coordinates = 0.0
            if 0.0 in point:
                temp_sign_list = []
                temp = []
                
                # Find 0 indices 
                zero_indices = [i for i,e in enumerate(point) if e == 0.0]
                
                # Generate sign combinations for all the non_zero elements
                temp_sign_list = generate_signs(dim-len(zero_indices))
                
                # Generate the signed solutions as vectors that do not include zero elements
                for sign_combination in temp_sign_list:
                    temp.append([-x if y == 0 else x*y for x,y in zip([entry for entry in point if entry != 0.0],sign_combination)])
                
                # Reinsert the zeros back into the array to obtain the correct solution
                # TODO: Could we do this more efficiently, just with one list comprehension?
                for zero_free_soln in temp:
                    for index in zero_indices:
                        zero_free_soln.insert(index,0.0)
                    solutions.append(zero_free_soln)
            else:
                # The signs combination is represented as a binary number where 0 is - and 1 is + 
                for sign_combination in signs_list:
                    solutions.append([-x if y == 0 else x*y for x,y in zip(point,sign_combination)])
        return solutions    
        
    
    def generate_l2_lattice(self,dim=2,radius=1,lower_bound=-1,upper_bound=1,num_points=5,pos_ord=True,rel_tol=1e-06):
        """ Generates a lattice inside the dim-dimensional hypersphere of radius @radius. The lattice is symmetric and its 
        coordinates are in the interval [-@lower_bound,@upper_bound]. The [-@lower_bound,@upper_bound] is discretised in to 
        @num_points (including the edges). If @pos_ord is True, then all the permutations and signed solutions derived from the 
        positive solutions with the ordering x1>=x2>=...>=xd are also returned"""
        
        # Set the relative tolerance for assertions and radius updates checks
        self.rel_tol = rel_tol
        
        # Find lattice coordinates
        full_lattice_coord = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True) # Removed precision
        
        # Extract positive coordinates
        pos_lattice_coord = list(full_lattice_coord[full_lattice_coord >= 0.0])
        
        # Compute all the solutions with all fixed x_1 > sqrt(r^2/d)
        # self.points.extend(self.main_recursion(radius,dim,pos_lattice_coord,lb=math.sqrt(radius/dim),ub=radius,x_prev = [],max_dim=dim))
        self.main_recursion(radius,dim,pos_lattice_coord,lb=math.sqrt(radius/dim),ub=radius,x_prev=[],max_dim=dim)
        # Compute all d-dimensional solutions with x1>=x2>=....>=xd and x_i <= sqrt(r^2/d) for all i \in d
        self.points.extend(self.ordered_recursion(pos_lattice_coord,radius,dim,math.sqrt(radius/dim)))
        
        # Generate signed and permuted solutions
        if pos_ord:
            self.points = self.generate_permuted_solutions(self.points)
            self.points = self.generate_signed_solutions(self.points,dim)
        
        # TODO: make pos_lattice_coord a property of the object? This would seem to make sense.
        # TODO: Are lines 90-93 necessary for the algorithm? A: Should be, but a better understanding of the scopes during recursion would be nice
        # TODO: I think a slight improvement could be made if I remeoved the reversed() in line 49 and used break instead of continue - would this
        # work correctly or would affect the recursion. Early versions of the algo had break but didn't work.

