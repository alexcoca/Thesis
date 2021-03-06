# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:53:37 2018

@author: alexc
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import mlutilities as mlutils
# import Errors 

class DiscreteGenerator():
    def __init__(self,reg_slope=1,reg_intercept=0,num_pts=20,frac_test=0.2,num_x_locs=20,\
                 x_locs=[],y_locs=[],dom_x_bound=1,points_per_loc=1,num_pts_x_lattice=20,\
                 num_pts_y_lattice=50,batch_size=20):
        ''' Initialisation of DataGenerator object 
        
        @ param: set_type: indicates if a regression or classification data set is to be created
        @ type: string: 'classification','regression' 
        
        @ param dimensionality: feature dimensionality
        @ type: strictly positive integer
        
        @ param num_pts_x_lattice: number of points used to define the x coordinate of the lattice
        
        @ param num_pts_y_lattice: number of points used to defnine the y coordinate of the lattice
        
        @ param reg_slope: regression line slope
        
        @ param reg_intercept: regression line intercept
        
        @ param domain_x_bound: upper bound of the domain over which the lattice is defined (x direction)
        lower bound is assumed to be -domain_x_bound. The regression line is restricted to the same domain.
        
        @ param domain_y_bound: upper bound of the domain over which the lattice is defined (y direction)
        
        @ param num_pts. Total number of points to be generated. This number should be smaller or equal to the 
        product of the lengths of the x and y coordinates
        
        @ param frac_test. Fraction (of the total number of points) that will be used for testing
        
        @ param num_x_locs: The number of distinct x-axis locations where data points are to be generated. These are selected
        randomly from the lattice x coordinates. If set to zero, the x coordinates at which the points are generated are passed 
        through the x_locs vector:
            
        @ param x_locs: a vector whose entries define the x coordinates of the data points to be returned. Used if @param num_x_locs
        is set to 0
        
        @ param points_per_loc: an integer number specifying how many different points are to be generated with the same x location. An 
        even number of points is alocated to all the x_locations, if num_points is divisible by points_per_loc. Otherwise, the remainder 
        to the division num_points/points_per_loc is allocated randomly across the available x_coordinates
        
        @ param min_dist: the closest data point to the regression line is at a distance > min_dist * lattice step size in y direction
        
        @ param max_dist: the farthest data point to the regression line is at a distance < max_dist * lattice step size in y direction
        
        @ param batch_size: If the data set is required to have more points than lattice x coordinates, then batches of points of size
        batch_size are sampled in turn until the required number of points is sampled
        
        @ param data: array containing the data 
        '''
        self.set_type = 'regression'
        self.dimensionality = 1 # TODO: Not used for the time being - only 1-D sets are generated
        self.num_pts_x_lattice = num_pts_x_lattice
        self.num_pts_y_lattice = num_pts_y_lattice
        self.reg_slope = reg_slope
        self.reg_intercept = reg_intercept
        self.num_pts = num_pts
        self.frac_test = frac_test # TODO: Not implmented, used to return a test set
        self.num_x_locs = num_x_locs # TODO: Not implemented, used to control how many x locations
        self.current_x_locs = []
        self.x_locs = x_locs
        self.y_locs = y_locs
        self.dom_x_bound = dom_x_bound
        # self.points_per_loc = points_per_loc #TODO: Not implemented. Controls max number of points at a given coordinate
        self.min_dist_fact = 0.5
        self.max_dist_fact = 5
        self.num_dec = 2
        self.lattice = {}
        self.regression_line = np.zeros((num_pts_x_lattice,2))
        self.dom_y_upper_bound = 0
        self.dom_y_lower_bound = 0
        self.batch_size = batch_size
        np.random.seed(seed=5)
        
    def generate_reg_line (self,reg_slope,reg_intercept): 
        ''' This method returns a vector containing a regression line specified 
        over [-dom_x_bound,dom_x_bound], defined by parameters reg_slope and 
        reg_intercept. Used for visualisation purposes, not needed for data gen.
        Parameters:
            @reg_slope: slope of the line (float)
            @reg_intercept: intercept of the line (float)
            @dom_x_bound: specifies the upper limit of the interval over which the line is defined (float). 
            Interval is always symmetric so the lower bound is the negative of the number provided (float)
            @num_pts_x_lattice: specifies how fine the discretisation '''
        num_dec = 2
        self.reg_slope = reg_slope
        self.reg_intercept = reg_intercept
        x_vals = np.linspace(-self.dom_x_bound,self.dom_x_bound,self.num_pts_x_lattice,endpoint=True)
        y_vals = reg_slope*x_vals+reg_intercept
        line = np.zeros((np.shape(x_vals)[0],2))
        line[:,0] = np.round(x_vals,num_dec)
        line[:,1] = np.round(y_vals,num_dec)
        self.regression_line = line
        
    
    def generate_lattice (self,num_pts_x_lattice,num_pts_y_lattice):
        ''' This method returns a lattice over x and y coordinates
        Parameters:
            @ num_pts_x: number of coordinates in the x direction
            @ num_pts_y: number of coordinates in the y direction
            @ y_upper_bound: upper bound for the lattice domain in the y direction
            @ y_lower_bound: lower bound for the lattice domain in the y direction
            '''
        x_vals = np.linspace(-self.dom_x_bound,self.dom_x_bound,num_pts_x_lattice,endpoint=True)
        x_vals = np.round(x_vals,decimals=self.num_dec)
        y_lower_bound = -self.reg_slope*self.dom_x_bound+self.reg_intercept
        y_upper_bound = np.ceil(self.reg_slope*self.dom_x_bound+self.reg_intercept)
        y_vals = np.linspace(y_lower_bound,y_upper_bound,num_pts_y_lattice,endpoint=True)
        y_vals = np.round(y_vals,decimals=self.num_dec)
        lattice = {'x_vals':x_vals,'y_vals':y_vals}
        self.lattice = lattice
        self.dom_y_upper_bound = y_upper_bound
        self.dom_y_lower_bound = y_lower_bound
    
    def plot_lattice(self):
        ''' This method plots the lattice specified by the coordinates of its x and y values'''
        # Create an array with the coordinates of all the points in the lattice
        points = np.zeros((self.num_pts_x_lattice*self.num_pts_y_lattice,2))
        cnt = 0 
        for x_coord in self.lattice['x_vals']:
            points[self.num_pts_y_lattice*cnt:self.num_pts_y_lattice*(cnt+1),0] = x_coord
            points[self.num_pts_y_lattice*cnt:self.num_pts_y_lattice*(cnt+1),1] = self.lattice['y_vals']
            cnt += 1
        # Plot lattice
        plt.scatter(points[:,0],points[:,1],marker='.')
                    
    def generate_data_set(self,num_pts,reg_slope,reg_intercept,num_pts_x_lattice,num_pts_y_lattice):
        ''' This method generates a synthetic data set for regression'''
        def sample_y_locations(max_dist):
            ''' This helper function samples a data point for each coordinate in self.x_locs ''' 
            for x in self.current_x_locs: 
            # Calculate point on regression line
                reg_point = np.round(self.reg_slope*x+self.reg_intercept,decimals=self.num_dec)
                # Calculate interval from which the y coordinates will be sampled
                lower_limit = np.max([reg_point - max_dist,self.dom_y_lower_bound])
                upper_limit = np.min([reg_point + max_dist,self.dom_y_upper_bound])
                # Retrive the y coordinates in the [lower_limit,upper_limit] interval
                y_idx = np.where(np.logical_and(self.lattice['y_vals'] >= lower_limit,self.lattice['y_vals'] <= upper_limit))
                y_range = self.lattice['y_vals'][y_idx]
                # Sample y location and append to the list of sampled points
                y_sample = np.random.choice(y_range,size=1,replace=False)
                self.y_locs = np.append(self.y_locs,y_sample)
        def sample_x_locations(num_pts,num_pts_x_lattice): 
            ''' This helper function chooses, uniformly at random, num_points from the lattice x coordinates''' 
            # self.current_x_locs is used  because we might need to call this function multiple times to obtain different sets of x coordinates
            # TODO: figure out why the following syntax gives you an array within an array?
            self.current_x_locs = np.round(np.sort(self.lattice['x_vals']\
                                   [np.random.choice(range(0,num_pts_x_lattice),size=(1,num_pts),replace=False)][0]),\
                                    decimals=self.num_dec)
            # All sampled x coordinates are stored in x_locs
            self.x_locs = np.append(self.x_locs,self.current_x_locs)
        def remove_duplicates():
            ''' This helper function should remove duplicates from the data and tries to replace them with 
            distinct data points ''' 
            pass
        # Generate regression line and lattice
        self.generate_reg_line(reg_slope,reg_intercept) # For visualisation 
        self.generate_lattice(num_pts_x_lattice,num_pts_y_lattice)
        # Calculate min distance from regression line
        lattice_y_step = self.lattice['y_vals'][1]-self.lattice['y_vals'][0]
        # TODO: Adjust max_dist to avoid sampling the same point - keep track of the points already sampled.
        # Could also handle by resampling a few points at the end if duplicates are detected in the final array
        # TODO: See if it is necessary to implement min_dist - this just ensures that the points would 
        # be at least min_dist from the regression samples
        min_dist = self.min_dist_fact*lattice_y_step 
        max_dist = self.max_dist_fact*lattice_y_step
        if not self.x_locs: # Randomly choose x locations if not specified
            if num_pts < num_pts_x_lattice:
                # Sample x locations
                sample_x_locations(num_pts,num_pts_x_lattice)
                sample_y_locations(max_dist)
            else: # More points than lattice coordinates
                # Repeatedly sample x and y locations
                no_sampling_steps = np.floor(num_pts/self.batch_size)
                remainder = num_pts % self.batch_size
                while (no_sampling_steps > 0):
                    sample_x_locations(self.batch_size,num_pts_x_lattice)
                    sample_y_locations(max_dist)
                    if no_sampling_steps == 1:
                        if remainder != 0:
                            sample_x_locations(remainder,num_pts_x_lattice)
                            sample_y_locations(max_dist)
                    no_sampling_steps -= 1
        else:
            # Here we just sample y locations since the x_locations are given to us
            # Again, need to handle the case where 
            pass
        # Store everything in one data structure
        self.data = np.zeros((num_pts, 2))
        self.data[:,0] = self.x_locs
        self.data[:,1] = self.y_locs
        # TODO: when more than one point is to be alocated at a specific coordinate, \
        # ensure that you have enough points in (min_di0st,max_dist) interval. If not double
        # max_dist_fact so that you get enough points. Make sure that regr_line_value + max_dist
        # falls below y_upper_bound 


class ContinuousGenerator():
    
    def __init__(self, d = 1, n = 1, mean = 0, variance = 1, perturbation = False, perturbation_mean = 0, perturbation_variance = 1, seed = 23):
        ''' Parameters:
            @ d (dimensionality): number of features
            @ n: number of data points
            @ perturbation: if True, Gaussian noise with @perturbation_mean
            and @perturbation_variance is added to the targets
            @mean, variance: mean and variance of the Gaussian that is used to generate the
            feature vector coordinates'''
        self.d = d
        self.n = n
        self.test_targets = []
        self.test_features = []
        self.test_data = []
        self.features = []
        self.targets = np.zeros(shape=(n,1))
        self.data = []
        self.coefs = []
        self.perturbation = perturbation 
        self.perturbation_variance = perturbation_variance
        self.variance = variance
        self.perturbation_mean = perturbation_mean
        self.mean = mean
        self.seed = seed
        
    def generate_data(self, seed = 23, test_frac = 0.0, bound_recs = True):
        ''' This function generates data on a hyperplane in R^d. 
        The coefficients (@coeff) are sampled at random from [-1,1].
        The domain points are sampled using a Gaussian distribution.
        If bound_recs is set to True, then a transformation is applied to the
        dataset such that the records have 2-norm <= unity.'''
        
        def get_indices(mask, n_rep):
            ''' Finds the indices of the feature vectors that yielded a target 
            >= target bound and returns a subset equal to the number of replacements that 
            can be made as well as mask, which reflects the updated state of mask after samples replacement'''
            
            # Find the indices of the feature vectors which yield target >= target_bound
            good_values = [False]
            
            # Choose the first n_rep indices corresponding to features that generate targets >= target_bound
            indices = np.where(np.isin(mask, good_values))[0][:n_rep]
            
            # Update the mask to reflect the fact that some of the features might have been updated
            mask[indices] = True
            
            return indices, mask

        self.seed = np.random.seed(seed)
        upper_bound = 1
        lower_bound = -1
        target_bound = 1
        
        # Sample coefficients
        self.coefs = (upper_bound - lower_bound)*np.random.random((self.d, 1)) + lower_bound

        # Sample features and normalise them s.t. their 2-norm is <=1
        self.features = np.random.normal(loc = self.mean, scale = self.variance, size = (self.n, self.d))
        if bound_recs == True:
            self.features, normalizer = mlutils.bound_records_norm(self.features)
            # y_idx = np.where(np.logical_and(self.lattice['y_vals'] >= lower_limit,self.lattice['y_vals'] <= upper_limit))

        # Generate targets restricted to [-target_bound,target_bound]
        temp_targets = np.sum(self.coefs.T*self.features, axis = 1, keepdims = True)
        
        # Add noise to the targets 
        if self.perturbation:
            temp_targets = temp_targets + np.random.normal(loc = self.perturbation_mean, scale = self.perturbation_variance,\
                                                           size = temp_targets.shape)
        
        # Check if all targets are within the bounds 
        if np.all(np.abs(temp_targets) <= target_bound):
            self.targets = temp_targets
        else:
            # Determine how many points need to be resampled so that all feature vectors yield targets within the desired bound
            mask = np.abs(temp_targets) <= target_bound
            self.targets[mask] = temp_targets[mask]
            resample_no = len(temp_targets[np.logical_not(mask)])
            while resample_no > 0:
                # Resample solutions and calculate their corresponding targets
                proposed_features = np.random.normal(loc = self.mean, scale = self.variance, size = (resample_no, self.d))
                if bound_recs == True:
                    max_norm = np.max(np.linalg.norm(proposed_features, ord = 2, axis = 1)) 
                    if max_norm  < normalizer:
                        # Normalise the proposed features
                        proposed_features = proposed_features / normalizer
                    else:
                        proposed_features = proposed_features / max_norm
                proposed_targets = np.sum(self.coefs.T*proposed_features, axis = 1, keepdims = True) 
                if self.perturbation:
                    proposed_targets = proposed_targets + np.random.normal(loc = self.perturbation_mean, scale = self.perturbation_variance, \
                                                                           size = proposed_targets.shape)
                
                # How many good targets have we sampled 
                good_targets = proposed_targets[np.abs(proposed_targets) <= target_bound]
                good_features = proposed_features[np.ravel(np.abs(proposed_targets) <= target_bound)]
                resample_no = resample_no - len(good_targets)
                # Replace the feature vectors which yield targets >= target_bound by feature vectors which yield targets <= target_bound
                rep_indices, mask = get_indices(mask,len(good_targets))
                if good_features.size > 0:
                    self.features[rep_indices,:] = good_features
                    # Replace the old targets with new ones that are within the prescribed bound
                    self.targets[rep_indices,:] = good_targets.reshape(good_targets.shape[0],-1)

        # Slice arrays to create test data
        if test_frac > 0.0:
            slice_arr  = np.arange(self.features.shape[0] - self.n*test_frac, self.features.shape[0], dtype = int)
            self.test_features = self.features[slice_arr,:]
            self.test_targets = self.targets[slice_arr,:]
            self.features = np.delete(self.features, slice_arr, axis = 0)
            self.targets = np.delete(self.targets, slice_arr, axis = 0)
            self.data = np.concatenate((self.features, self.targets), axis = 1)
            self.test_data = np.concatenate((self.test_features, self.test_targets), axis = 1)
        else:
            self.data = np.concatenate((self.features, self.targets), axis = 1)
        
    def plot_data(self, angle_1 = 30, angle_2 = 0, add_hyperplane = False, hyperplane = []):
        ''' Plot generated data if the dimensionality of the data is one'''
        if self.d == 1:
            plt.plot(self.features, self.targets, 'b*')
            plt.plot(self.test_features, self.test_targets, 'r*')
        if self.d == 2:
            num_pts = 50
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            # Create grid 
            x_coord = np.linspace(np.min(self.features[:, 0]), np.max(self.features[:, 0]), num = num_pts, endpoint = True)
            y_coord = np.linspace(np.min(self.features[:, 1]), np.max(self.features[:, 1]), num = num_pts, endpoint = True)
            xx, yy = np.meshgrid(x_coord,y_coord)
            # Evaluate the function on the grid
            z = self.coefs[0]*xx + self.coefs[1]*yy
            ax.plot_surface(xx, yy, z, alpha = 0.2)
            if add_hyperplane:
                x_coord_add = np.linspace(np.min(self.test_features[:, 0]), np.max(self.test_features[:, 0]), num = num_pts, endpoint = True)
                y_coord_add = np.linspace(np.min(self.test_features[:, 1]), np.max(self.test_features[:, 1]), num = num_pts, endpoint = True)
                xx_add, yy_add = np.meshgrid(x_coord_add, y_coord_add)
                z = hyperplane[0]*xx + hyperplane[1]*yy
                ax.plot_surface(xx_add, yy_add, alpha = 0.5)
            ax.scatter(self.features[:, 0], self.features[:, 1], self.targets[:], 'b*')
            ax.scatter(self.test_features[:, 0], self.test_features[:, 1], self.test_targets[:], 'r*')
            ax.view_init (angle_1, angle_2)
            ax.set_xlabel(r'$x_1$', fontsize = 20)
            ax.set_ylabel(r'$x_2$', fontsize = 20)
            ax.set_zlabel(r'$y$', fontsize = 20)
            for angle in range(0, 360):
                ax.view_init(30, angle)
                plt.draw()
                plt.pause(.001)    


        else:
            # TODO: Error handling 
            pass

    