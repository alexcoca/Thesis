# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:03:44 2018

@author: alexc
"""
import numpy as np
import operator
import itertools


def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1,filtered=False,num_dec=5):
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
    coord_array = list(np.round(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True),decimals=num_dec))
    lattice_coordinates = [coord_array for i in range(dim)]
    
    # Generate vectors on the hypercube
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    
    # Generate lattice
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    if filtered:
        tmp = filter_unsorted(intersection)
        intersection = filter_signed(tmp)
        
    return (intersection,coord_array)   

def get_differences(list_a=[],list_b=[]):
    ''' Returns those elements that are in list_a but not in list_b'''
    #TODO: do this without casting to list, it's not efficient
    if isinstance(list_a,np.ndarray):    
        list_a = [list(element) for element in list_a]
    if isinstance(list_b,np.ndarray):
        list_b = [list(element) for element in list_a]
    return [x for x in list_a if x not in list_b]

        


