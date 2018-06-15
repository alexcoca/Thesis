# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:03:44 2018

@author: alexc
"""
import numpy as np 

def bruteNonIntegerIntersection(dim,radius,num_points=5,lower_bound=-1,upper_bound=1):
    coord_array = list(np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True))
    lattice_coordinates = [coord_array for i in range(dim)]
    flat_grid = np.array(np.meshgrid(*lattice_coordinates)).T.reshape(-1,dim)
    intersection = flat_grid[np.linalg.norm(flat_grid,ord=2,axis=1) <= radius] 
    return (intersection,coord_array) 