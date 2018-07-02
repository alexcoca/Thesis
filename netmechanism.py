# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 11:51:56 2018

@author: alexc
"""
import math
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import mlutilities as mlutils
import itertools
import os, pickle, glob
from scipy.special import comb, factorial
from multiprocessing import Pool
import utility_functions
import functools
from netmechanism_helpers import FileManager

class FeaturesLattice():
    
    '''A FeaturesLattice object contains, in the @points property, all vectors with 
    norm less than @radius of dimensionality @dim. The coordinates of the vectors lie on a lattice
    parametrised by its upper and lower limits, and the number of points in which the interval bounded
    by them is discretised. The lattice is always symmetric.'''
    
    def __init__(self):
        self.dim = 2 
        self.radius = 1
        self.points = []

    def save_lattice(self,folder_path,lattice_name):
        ''' Saves the lattice to the location specified by @folder_name with
        the name specified by @lattice_name.'''
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            
        full_path = folder_path+"/"+lattice_name
       
        # Raise an error if the target file exists
        if os.path.exists(full_path):
            assert False
           
        with open(full_path,"wb") as data:
            pickle.dump(self.points,data)
    
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
        
        self.points = np.array(self.points)
        
        # TODO: make pos_lattice_coord a property of the object? This would seem to make sense.
        # TODO: Are lines 90-93 necessary for the algorithm? A: Should be, but a better understanding of the scopes during recursion would be nice
        # TODO: I think a slight improvement could be made if I remeoved the reversed() in line 49 and used break instead of continue - would this
        # work correctly or would affect the recursion. Early versions of the algo had break but didn't work.

class TargetsLattice():
    
    '''A TargetsLattice object contains all vectors of dimension d, whose individual 
    entries are chosen from a discrete set of num_points points defined on the closed
    [lower_bound,upper_bound] interval. All d! permutations are also included. The vectors
    are stored in the .points property as a numpy array'''
    
    def __init__(self):
        self.points = []
        
    def save_lattice(self,folder_path,lattice_name):
        ''' Saves the lattice to the location specified by @folder_name with
        the name specified by @lattice_name.'''
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            
        full_path = folder_path+"/"+lattice_name
       
        # Raise an error if the target file exists
        if os.path.exists(full_path):
            assert False
           
        with open(full_path,"wb") as data:
            pickle.dump(self.points,data)
    
    def generate_lattice(self,dim=2,lower_bound=-1,upper_bound=1,num_points=5):
        ''' Generates the lattice coordinate vector, the combinations of d points and their permuations. '''
        # Necessary since we choose combinations of d elements from num_points
        self.dimensionality = dim
        # TODO: error handling, case d > num_points
        
        def generate_permutations(points):
            """"Generates all the permutations of the solutions contained in the
            points attribute"""
            
            solutions = []
            
            for point in self.points:
                    solutions.extend(list(itertools.permutations(point,self.dimensionality)))
            return solutions
        
        # Generate 1-d lattice 
        full_lattice_coord = np.linspace(lower_bound,upper_bound,num=num_points,endpoint=True) 
        
        # Generate all combinations of subsets of size dim from the lattice coordinate array.
        # These correspond to all possible assignments of (w_1,...,w_d)
        combinations_idx = mlutils.findsubsets(range(len(full_lattice_coord)),self.dimensionality)
        
        # Generate all sets of targets
        for combination_ids in combinations_idx:
            self.points.append(full_lattice_coord[combination_ids])
        
        # Permute all sets of generated targets
        self.points = np.array(generate_permutations(self.points))                  
            
class OutcomeSpaceGenerator(FileManager):        
    
    def __init__(self, directory = '', experiment = '', synth_features = [], synth_targets = [], private_data = [],\
                 property_preserved = 'second_moments', privacy_constant = 0.1, batch_size = 100, parallel = False,\
                 partition_method = 'fast'):
        ''' @ parallel: specifies whether the calculations of the outcome space scores is to be 
            performed in parallel or not
            @ partition_method: Specifies which implementation is to be used to calculate the partition function'''
       
        # Initialise FileManager class
        super(OutcomeSpaceGenerator, self).__init__()
        
        # Determines whether execution happens in parllel or not
        self.parallel = parallel
        self.partition_method = partition_method
        
        # User defined properties/ inputs
        self.directory = directory 
        self.experiment_name = experiment
        self.synth_features = synth_features
        self.synth_targets = synth_targets
        self.private_data = private_data
        self.property_preserved = property_preserved
        self.epsilon = privacy_constant
        self.batch_size = batch_size
        self.dimensionality = 2
        self.scaling_const = 0
        
        # Contains the filenames of scaled utilities on disk. Set by the 
        # generate_outcome and generate_outcomes_parallel methods
        self.filenames = [] 
        
        self.n_batches = 0
    
        # Calculate private data utility
        self.F_tilde_x = 0
    
        # Results storage    
        self.batch_results = [] # Stores the max_score and the matrix containing the scores for that batch
        self.partition_function = 0
        self.max_scaled_utility = 0
        
        
    def get_private_F_tilde (self, private_data):
        '''Compute F_tilde (equation (4.1), Chapter 4, Section 4.1.1)
        for the private data '''
        
        const = (1/self.private_data.features.shape[0])
        F_x = const*self.private_data.features.T@self.private_data.features
        f_x = const*self.private_data.features.T@self.private_data.targets
        F_tilde_x = np.concatenate((F_x,f_x), axis = 1)
        
        return F_tilde_x 
    
    def save_batch(self, batch, filename, directory = '', overwrite = False):
        ''' Saves the batch of scores to the location specified by @directory with
        the name specified by @filename.'''
        
        # If directory is not specified, then the full path is provided in filename
        # This is used to overwrite the files containing the raw scores during the 
        # calculation of the partition function to avoid unnecessary duplication of 
        # operations during the sampling step
        
        if not directory:
            full_path = filename
            
            if overwrite:
                with open(full_path, "wb") as data:
                    pickle.dump(batch,data)
            else:
                # Overwriting data only alowed if this is explicitly mentioned
                if os.path.exists(full_path):
                    assert False
        else:
            # Create directories if they don't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            full_path = directory + "/" + filename
           
            # Raise an error if the target file exists
            if os.path.exists(full_path):
                    raise IOError ("File already exists, delete before continuing!")
               
            with open(full_path,"wb") as data:
                pickle.dump(batch, data)
                              
    def evaluate_sample_score(self, batch_index):
        
        if self.scaling_const == 0:
            raise ValueError("Scaling constant has not been calculated!")
    
        # Storage structure
        struct = {}
        
        # Store the batch index to be able to retrieve the correct sample during sampling step
        struct['batch_index'] = batch_index
        
        # Generate a batch of combinations according to the batch_index
        batch = itertools.islice(itertools.combinations(range(self.synth_features.shape[0]),self.dimensionality), \
                                     (batch_index)*self.batch_size,(batch_index+1)*self.batch_size)
        
        # Evaluate utility - note that exponential is not taken as sum-exp trick is implemented to 
        # evalute the scores in a numerically stable way during the sampling stage
        if self.property_preserved == 'second_moments':
            scaled_utilities_batch = utility_functions.compute_second_moment_utility(self.synth_targets, self.synth_features[list(batch),:],\
                                                                                     self.dimensionality, self.scaling_const, self.F_tilde_x)
        else:
            scaled_utilities_batch = utility_functions.compute_first_moment_utility(self.synth_targets, self.synth_features[list(batch),:],\
                                                                                     self.dimensionality, self.scaling_const, self.F_tilde_x)
        struct ['scaled_utilities'] = scaled_utilities_batch
        
        # Max. utility is calculated in order to implement exp-normalise trick
        max_scaled_util = np.max(scaled_utilities_batch)
        
        # Save scaled utilities to disk
        filename = self.experiment_name +  "_" + str(batch_index)
        self.save_batch(struct, filename, self.directory)
        
        # WARNING: Returning max_scaled_util is used for exp-normalise trick. score_batch allows
        # one to calculate the partition function fast but might cause memory errors
        # if many large batches are returned. 
        
        return (max_scaled_util,scaled_utilities_batch)
    
    def calculate_partition_function(self, results = [], partition_method = 'fast'):
        ''' Calculates the partition function. If @method is 'fast' then the partition function is 
        calculated from the iterable @results returned by the TBD method. If method is 'slow', then 
        the partition function is calculated by loading the data saved by the @evaluate_sample_score
        method during utility calculation.
        
        Notes: If @method is set to 'slow', then each of the scaled utility matrices saved on the disk is
        transformed by subtracting the @max_scaled_utility (sum-exp trick) and taking the element-wise
        matrix eponential, in preparation for the sampling step. This is NOT the case if the fast calculation
        is used. The fast calculation is possible for small cases, where all the scores matrices and the max
        scores can be kept in memory '''
    
        def get_batch_id(filename):
            ''' Helper function that processes a file name to obtain the batch index. '''
            return int(filename[filename.rfind("_") + 1:])
        
        def func(iterable, max_scaled_utility):
            ''' Helper function that calculates the partition function given an iterable
            @iterable which contains tuples of the form (scaled_utilities_batch, max_batch_scaled_util).''' 
            return np.sum(np.exp(iterable[1]-max_scaled_utility))
        
        if partition_method == 'slow':
            
            # Raise an error if max_scaled utility is positive. Default is positive to that
            # an error is raised if the utility is not calculated and the method is set to 'slow'
            if self.max_scaled_utility > 0:
                raise ValueError ("Maximum scaled utility incorrectly calculated!")
            if not self.filenames:
                raise RuntimeError ("Expected a list of filenames to load the scaled utilities!")
         
            # Filenames sorted by batch_index to ensure correct files are accessed during 
            # reloading
            filenames = sorted(self.filenames, key = get_batch_id)
    
            partition_function  = 0
            
            for batch in range(self.n_batches):
                
                # Retrive data for the corresponding batch
                data = self.retrieve_scores(filenames,[batch])[0]
                    
                # Apply sum-exp trick when calc. partition to avod numerical issues
                data['scaled_utilities'] = np.exp(data['scaled_utilities'] - self.max_scaled_utility)
                partition_function += np.sum(data['scaled_utilities'])
                
                # Overwrite the raw batch scores with the scores calculated as per 
                # Step 3 of the procedudre in Chapter 4, Section 4.1.3 
                self.save_batch(data,filenames[batch], overwrite = True)
            
            self.partition_function = partition_function
            
            print ("Partition function is", str(partition_function))
                
        if partition_method == 'fast':
            
            # Raise an error if the results stucture is not provided and the method  arg is set
            # to 'fast'
            if not results:
                raise RuntimeError ("Results iterable not provided!")
            
            # Create a copy of the results iterable
            iter_1, iter_2 = itertools.tee(results)
            
            # Compute maximum scaled utility across all batches for exp-normalise trick
            max_scaled_utility = functools.reduce((lambda x,y:(max(x[0],y[0]),np.zeros(shape=y[1].shape))),iter_1)[0]    
    
            # Compute partition function
            partition_function = sum(map(functools.partial(func,max_scaled_utility = max_scaled_utility),iter_2))     
            
            self.partition_function = partition_function
            self.max_scaled_utility = max_scaled_utility
            
            print ("Partition function is", str(partition_function))
            print ("Max_scaled_utility is", max_scaled_utility)
        
    def generate_outcomes(self, experiment_name = '', synth_features = [], synth_targets = [],\
                          private_data = [], property_preserved = 'second_moments', privacy_constant = 0.1):        
            
        # Set object properties
        self.experiment_name = experiment_name
        self.directory = self.directory + "/" + str(experiment_name) + "/OutcomeSpace"
        self.synth_features = synth_features
        self.synth_targets = synth_targets
        self.private_data = private_data
        self.property_preserved = property_preserved
        # This is the inverse global sensitivity times the privacy parameter
        if self.property_preserved == 'second_moments':
            self.scaling_const = self.epsilon*self.private_data.features.shape[0]/(2*2)
        else:
            raise NotImplementedError
            # TODO: Implement alternatives
        self.epsilon = privacy_constant
        self.dimensionality =  self.private_data.features.shape[1]   
        self.n_batches = math.ceil(comb(self.synth_features.shape[0],\
                                        self.dimensionality, exact=True)/self.batch_size)
        self.F_tilde_x = self.get_private_F_tilde(private_data)
        
        print("Number of batches is",self.n_batches)
        
        if self.n_batches == 0:
            raise ValueError ("Number of batches cannot be 0!")
        
        # Results container
        results = []
        
        if self.parallel == False:
            
            # Calculate and store scaled utilities for all outcomes, which are split in n_batches
            for batch_index in range(self.n_batches):
                results.append(self.evaluate_sample_score(batch_index))
            
            # Filenames where the scaled utilities are stored - necessary for sampling step
            self.filenames = glob.glob(self.directory + "/*")
            
            # Compute the partition function 
            
            if self.partition_method == 'fast':
                self.calculate_partition_function(results = results, partition_method = self.partition_method)
            else:
                # Calculate the maximum scaled utility for sum_exp trick
                max_scaled_utility = - math.inf
                for result in results:
                    if result[0] > max_scaled_utility:
                        max_scaled_utility = result[0]
                self.max_scaled_utility = max_scaled_utility
                print ("Max_scaled_utility is", max_scaled_utility)
                self.calculate_partition_function(partition_method = self.partition_method)
        else:
            self.generate_outcomes_parallel()
    
    def generate_outcomes_parallel():
        # TODO: make sure filenames is set correctly so the Sampler still works 
        raise NotImplementedError ("Parallel computations not implemented!")
    
class Sampler(FileManager):
    def __init__(self, directory = '', filenames = [], num_samples = 5, n_batches = 0, partition_method = 'fast', seed = 23,\
                 samples_only = False,  sampling_parameters = {}):
        
        # Initialise FileManager class
        super(Sampler, self).__init__()
        
        # Properties that should be set explicitly during initialisation
        self.num_samples  = num_samples
        self.partition_method = partition_method
        
        # If more samples are required, the seed is changed so that different
        # samples are returned
        if samples_only == False:
            self.seed = seed
        else:
            self.seed = seed + 1
            
        self.samples_only = samples_only
        self.sampling_parameters = sampling_parameters
        
        # Properties set by the sample method
        self.directory = ''
        self.filenames = filenames
        self.n_batches = 0
        self.batch_size = 0
        self.partition_function = 0
        self.max_scaled_utility = 1
        self.synth_features = []
        self.synth_targets = []
        self.dimensionality  = 2
        
        # If the Sampler() is instanstiated just to draw more samples for a 
        # previous experiment, the sampling parameters are passed to the object
        # using the @parameters dictionary, which is used for initialisation
        if samples_only:
            self.unpack_arguments()
        
        # Containers
        self.sample_indices = []
        self.sampled_data_sets = []
        
    def unpack_arguments(self):
        self.directory = self.sampling_parameters['directory']
        self.batch_size = self.sampling_parameters['batch_size']
        self.n_batches = self.sampling_parameters['n_batches']
        self.partition_function = self.sampling_parameters['partition_function']
        self.filenames = self.sampling_parameters['filenames']
        self.max_scaled_utility = self.sampling_parameters['max_scaled_utility']
        self.synth_features = self.sampling_parameters['synth_features']
        self.synth_targets = self.sampling_parameters['synth_targets']
        self.dimensionality  = self.sampling_parameters['dimensionality']
    
    def pack_arguments(self):
        self.sampling_parameters = {'directory': self.directory,'batch_size': self.batch_size,'n_batches': self.n_batches,\
                            'partition_function': self.partition_function, 'filenames': self.filenames,\
                            'max_scaled_utility':self.max_scaled_utility, 'synth_features': self.synth_features,
                            'synth_targets': self.synth_targets,'dimensionality': self.dimensionality}
        
    def sample_datasets(self, n_batches, num_samples, filenames, partition_function):
        ''' This method samples @num_samples data sets from the space generated by the
        OutcomeSpaceGenerator object. The locations of the files containing unnormalized
        probabilities (scores) of the outcomes are specified in the @filenames list. 
        The partition function is calculated by the OutcomeSpaceGenerator.'''
        
        # Set random number generator for repeatablity 
        np.random.seed(self.seed)
        
        def get_sample(scaled_partition):
            ''' This function subtracts the batch cumulative scores from the 
            (scaled) partition function until the partition function becomes negative. 
            When this occurs, the batch index is stored and a call is made 
            to the get_sample_idxs function to determine the entry in the matrix
            that corresponds to the zero crossing of the scaled partition function.'''
            
            def get_sample_idxs(scores,scaled_partition):
                ''' This function returns the row (row_idx) and column (col_idx)
                index of the entry in the scores matrix for which the partition function
                becomes negative'''
                
                row_idx = 0      
                col_idx = 0 
                
                # Calculate cumulative scores for each batch
                cum_scores = np.sum(scores, axis = 1)
                candidate_partition = scaled_partition
                
                # Step 1: Subtract the batch cumulative scores until scaled partition function
                # becomes negative. Remember the smallest positive value
                while candidate_partition > 0:
                    candidate_partition = scaled_partition - cum_scores[row_idx]
                    if candidate_partition > 0:
                        scaled_partition = candidate_partition 
                        row_idx += 1
                
                # Subtract the entries of the row in which the partition function became
                # negative at Step 1 from the smallest postive value, stopping when the
                # scaled partion becomes negative
                for element in scores[row_idx,:]:
                    scaled_partition -= element
                    if scaled_partition > 0:
                        col_idx += 1
                    else:
                        break
                return (row_idx,col_idx)
            
            # Retrive the saved scores and subtract from the scaled partition function 
            # until it becomes negative. The data corresponding to the score for which
            # this zero crossing occurs is the sampled data set
            
            orig_partition = scaled_partition
            
            for batch in range(n_batches):
                
                # For the 'slow' mode, the sum-exp trick and exponentiation have been applied
                scores = self.retrieve_scores(filenames,batches=[batch])[0]['scaled_utilities']
               
                # Perform sum_exp trick and exponentiate if the fast method has been used for 
                # partition calculations
                if self.partition_method == 'fast':
                    scores = np.exp(scores - self.max_scaled_utility)
                    
                candidate = scaled_partition - np.sum(scores)
                if candidate > 0:
                    scaled_partition = candidate
                else: 
                    row_idx, col_idx = get_sample_idxs(scores,scaled_partition)
                    break
            return (batch, row_idx, col_idx, orig_partition)
       
        def get_batch_id(filename):
            return int(filename[filename.rfind("_") + 1:])
            
        # Store sample indices
        sample_indices = []
         
        # Filenames have to be sorted to ensure correct batch is extracted
        filenames  = sorted(filenames, key = get_batch_id)
        
        for i in range(num_samples):
            
            # To sample, a random number in [0,1] is first generated and multiplied with the partition f
            # (Step 5, in Chapter 4, Section 4.1.3)
            scaled_partition = partition_function*np.random.random()
            # To get a sample, the scores from each batch are subtracted from the scaled partition
            # until the latter becomes negative. The data set for which this zero crossing is 
            # attained is the sampled value ( Step 6, Chapter 4, Section 4.1.3)
            sample_indices.append(get_sample(scaled_partition))
    
        self.sample_indices = sample_indices 

    def recover_synthetic_datasets(self, sample_indices):
        
        def nth(iterable, n, default=None):
            "Returns the nth item from iterable or a default value"
            return next(itertools.islice(iterable, n, None), default)
        
        # Data containers
        feature_matrices = []
        synthetic_data_sets = []
        
        # Batches the samples were drawn from
        batches = [element[0] for element in sample_indices]   
        
        # Combinations corresponding to the element which resulted in the zero crossing
        # These are used to recover the feature matrices
        combs_idxs = [element[1] for element in sample_indices]
        
        # List of indices of the target vectors for the sampled data sets
        target_indices = [element[2] for element in sample_indices]
        
        # Feature matrix reconstruction 
        for batch_idx, comb_idx in zip(batches, combs_idxs):
            
            # Reconstruct slice
            recovered_slice = itertools.islice(itertools.combinations(range(self.synth_features.shape[0]), self.dimensionality),\
                                               (batch_idx)*self.batch_size, (batch_idx+1)*self.batch_size)
            
            # Recover the correct combination 
            combination = nth(recovered_slice, comb_idx)
            print ("Recovered combination", combination)
        
            # Recover the feature matrix
            feature_matrices.append(self.synth_features[combination,:])

        # Reconstruct the targets for the synthethic feature matrix 
        for feature_matrix,target_index in zip(feature_matrices,target_indices):
            synthetic_data_sets.append(np.concatenate((feature_matrix, self.synth_targets[target_index,:].reshape(self.synth_targets.shape[1],1)), axis = 1))
            
        self.sampled_data_sets = synthetic_data_sets
        
    def sample(self, directory = '', filenames = [] , n_batches = 0, batch_size = 0, partition_function = 0, max_scaled_utility = 1, dimensionality = 2,\
               synth_features = [], synth_targets = []):
        
        # Set class properties
        if self.samples_only:
            self.unpack_arguments()
        else:
            self.directory = directory
            self.batch_size = batch_size
            self.n_batches = n_batches
            self.partition_function = partition_function 
            self.filenames = filenames 
            self.max_scaled_utility = max_scaled_utility
            self.synth_features = synth_features
            self.synth_targets = synth_targets
            self.dimensionality  = dimensionality 
            self.pack_arguments()
        
        self.sample_datasets(self.n_batches, self.num_samples, self.filenames, self.partition_function)
        self.recover_synthetic_datasets(self.sample_indices)

def est_outcome_space_size(N,d,k):
    '''This function estimates the size of the outcome space as a function of:
        @ N: Total number of vectors inside the d-dimensional sphere
          d: private data dimensionality
          k: Number of points in which the target interval is discretised '''
    return comb(N, d, exact = True)*comb(k, d, exact = True)*factorial(d, exact = True)#/10**7
    