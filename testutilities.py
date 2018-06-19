# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:03:44 2018

@author: alexc
"""
import numpy as np
import operator
import itertools
import math


def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1,filtered=False,num_dec=5,r_tol=1e-06):
    """ Generate a lattice inside the d-dimensional hypersphere. Brute force method,
    all coordinates are first generated on the d-dimensional hypercube and those outside the 
    hypershere discarded. 
    TODO: Params with explanations"""
    
    def filter_unsorted(array):
        """ Removes rows that are not sorted from a numpy array, @array """
        def is_sorted(iterable, reverse= True):
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
    #coord_array = list(np.round(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True),decimals=num_dec))
    coord_array = list(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True))
    lattice_coordinates = [coord_array for i in range(dim)]
    
    # Generate vectors on the hypercube
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    
    # Generate lattice
    # Original code
    # intersection = flat_grid[np.round(np.linalg.norm(flat_grid,ord=2,axis=1),decimals=num_dec) <= radius] 
    norms = np.linalg.norm(flat_grid,ord=2,axis=1)
    close_norms = [True if math.isclose(np.linalg.norm(x),1,rel_tol=r_tol) == True else False for x in norms]
    small_norms = [True if x <= radius else False for x in norms]
    indices = [x or y for x,y in zip(small_norms,close_norms)]
    intersection = flat_grid[indices]
    intersection_2 = flat_grid[np.round(np.linalg.norm(flat_grid,ord=2,axis=1),decimals=num_dec) <= radius]
    if filtered:
        tmp = filter_unsorted(intersection)
        intersection = filter_signed(tmp)
    
    # Sort according to the first column 
    intersection = intersection[intersection[:,0].argsort()[::-1]]
        
    return (intersection,coord_array)   

def get_differences_deprecated(list_a=[],list_b=[]):
    ''' Returns those elements that are in list_a but not in list_b'''
    #TODO: do this without casting to list, it's not efficient
    if isinstance(list_a,np.ndarray):    
        list_a = [list(element) for element in list_a]
    if isinstance(list_b,np.ndarray):
        list_b = [list(element) for element in list_a]
    return [x for x in list_a if x not in list_b]

def get_differences(array_a=[],array_b=[],rel_tol=1e-5):
    
    def in_array(candidate,array,rel_tol):
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
        if in_array(comparison,array_b-comparison,rel_tol):
                mask.append(False)
        else:
            mask.append(True)
    
    differences = array_a[mask]
    
    return differences
        

