# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:03:44 2018

@author: alexc
"""
import numpy as np
import operator
import itertools

def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1):
    coord_array = list(np.round(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True),decimals=5))
    lattice_coordinates = [coord_array for i in range(dim)]
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    return (intersection,coord_array) 


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
    pass

        


