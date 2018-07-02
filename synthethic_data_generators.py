# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:53:51 2018

@author: alexc
"""

from netmechanism import FeaturesLattice, TargetsLattice
import pickle

class SyntheticDataGenerator():
    
    def __init__(self, private_data, OutcomeSpace, Sampler = [], privacy_constant = 0.1, 
                 num_points_features = 8 , num_points_targets = 5 , \
                 feat_latt_path = '', target_latt_path = ''):
        ''' Parameters:
            @ private_data: an object containing the private data. Features are stored
            in the private_data.features and targets in private_data.targets.
            @ synth_data_type: a string indicating which type of 
            synthetic data set should be generated. Possible values include
            'second_moments', 'first_moments'
            @ epsilon: privacy parameter
            @ feat_latt_path: path to the lattice from which the synthethic feature
            vectors are drawn. If not specified a lattice is intiliased upon call 
            of the generate_data method
            @ target_latt_path: path to the lattice from which the synthethic target
            vectors are drawn. If not specified a lattice is intiliased upon call 
            of the generate_data method '''
        
        # Synthetic data properties
        self.property_preserved = 'second_moments' # overwritten by the generate_data method
        self.epsilon = privacy_constant
        self.synthetic_features = [] # These are the lattice feature vectors, not the sampled feature sets
        self.synthetic_targets = [] # These are combinations of lattice target scalars, not the sampled target vectors
        self.num_points_features = num_points_features # Used for features lattice initialisation
        self.num_points_targets = num_points_targets # Used for targets lattice initialisation
        
        # Outcome Space
        self.outcome_space = OutcomeSpace
        self.sampler = Sampler
        
        # Private data
        self.private_data = private_data
        
        # Data processing parameters
        self.dimensionality = self.private_data.features.shape[1]
        
        # Lattice initialisation
        self.feat_latt_path = feat_latt_path
        self.target_latt_path = target_latt_path
        self.initilise_lattices()
    
        # Storage for synthetic datasets
        self.synthetic_datasets = []
        # Storage for sampling parameters - this can be passed to a Sampler() object 
        # to gather more samples 
        self.sampling_parameters = {}
        
    def initilise_lattices(self):        
        
        ''' This method initialises the lattices which are used to construct the outcome space.
        Lattice paths for features and targets can be specified upon initialisation of a 
        NetMechanism object using the kwargs @feat_latt_path and @target_latt_path. If paths are not 
        specified, the method creates the lattices automatically at the beginning of the data generation 
        process becasuse the dimensionality (@dim) and the number of grid coordinates 
        (@num_points) are necessary to create them. Lattices are always defined over [-1,1] or inside the
        unit ball.
        '''      

        if not self.feat_latt_path:
            if self.num_points_features < 2:
                raise ValueError ("Incorrect feature lattice definition. Please provide feature lattice path \
                                  or set num_points_features to be an integer greater than 2.")                
            else:
                print ("Initialising synthetic feature space lattice")
                SyntheticFeatureSpace = FeaturesLattice()
                SyntheticFeatureSpace.generate_l2_lattice(dim = self.dimensionality, num_points = self.num_points_features)
                self.synthetic_features = SyntheticFeatureSpace.points
                print ("Synthetic feature space initialised")
        else:
            self.synthetic_features = self.load_lattice(self.feat_latt_path)
        if not self.target_latt_path:
            if self.num_points_targets < 2:
                raise ValueError ("Incorrect targets lattice definition. Please provide targets lattice path \
                                  or set num_points_targets to be an integer greater than 2.")
            else:
                print ("Initialising synthethic target space lattice")
                SynthethicTargetSpace = TargetsLattice()
                SynthethicTargetSpace.generate_lattice(dim = self.dimensionality, num_points = self.num_points_targets)
                self.synthetic_targets = SynthethicTargetSpace.points
                print ("Synthethic target space initialised")
        else:
            self.synthetic_targets = self.load_lattice(self.target_latt_path)
        
    def load_lattice(self,path):
        ''' Returns the contents of the file specified by absolute path '''
        with open(path,"rb") as data:
            lattice = pickle.load(data)
        return lattice

    def generate_data(self, property_preserved):
        
        # Set property preserved 
        self.property_preserved = property_preserved
        
        # Define experiment name
        experiment_name = "s_eps" + str(self.epsilon).replace(".","") + "d" +\
                                str(self.dimensionality) + "nt" + str(self.num_points_targets) +\
                                "nf" + str(self.num_points_features)
        
        # Generate outcomes and compute their scores
        self.outcome_space.generate_outcomes(experiment_name = experiment_name, synth_features = self.synthetic_features, \
                                             synth_targets = self.synthetic_targets, private_data = self.private_data,\
                                             property_preserved = self.property_preserved, privacy_constant = self.epsilon)
        
        # Sample the outcome space
        self.sampler.sample(directory = self.outcome_space.directory, filenames = self.outcome_space.filenames, n_batches = self.outcome_space.n_batches,\
                            batch_size = self.outcome_space.batch_size, partition_function = self.outcome_space.partition_function, \
                            max_scaled_utility = self.outcome_space.max_scaled_utility, dimensionality = self.dimensionality, synth_features = self.synthetic_features,\
                            synth_targets = self.synthetic_targets)
        
        # Return sampled values
        self.sampling_parameters = self.sampler.sampling_parameters
        self.synthetic_datasets = self.sampler.sampled_data_sets
        print ("The sampled datasets are",self.synthetic_datasets)
        
        
        
        



        