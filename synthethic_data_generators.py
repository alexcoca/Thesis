# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:53:51 2018

@author: alexc
"""

from netmechanism import FeaturesLattice, TargetsLattice
import pickle


class SynthethicDataGenerator():
    
    def __init__(self, synth_data_type, epsilon, OutcomeSpace, Sampler, feat_latt_path='',target_latt_path=''):
        ''' Parameters:
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
        self.synth_data_type = synth_data_type
        self.epsilon = epsilon
        self.synthetic_features = []
        self.synthetic_targets = []
        self.synthetic_data = []
        self.dimensionality = 2
        self.num_points = 4 # Used for lattice initialisation
        
        # Outcome Space
        self.outcome_space = OutcomeSpace
        self.sampler = Sampler
        
        # Private data
        self.private_data = []
        
        # Lattices
        self.feat_latt_path = feat_latt_path
        self.target_latt_path = target_latt_path
        
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
            print ("Initialising synthetic feature space lattice")
            SyntheticFeatureSpace = FeaturesLattice()
            SyntheticFeatureSpace.generate_l2_lattice(dim=self.dimensionality,num_points=self.num_points)
            self.outcome_features = SyntheticFeatureSpace.points
            print ("Synthetic feature space initialised")
        else:
            self.outcome_features = self.load_lattice(self.feat_latt_path)
        if not self.target_latt_path:
            print ("Initialising synthethic target space lattice")
            SynthethicTargetSpace = TargetsLattice()
            SynthethicTargetSpace.generate_lattice(dim=self.dimensionality,num_points=self.num_points)
            self.outcome_targets = SynthethicTargetSpace.points
            print ("Synthethic target space initialised")
        else:
            self.outcome_targets = self.load_lattice(self.target_latt_path)
        
    def load_lattice(self,path):
        ''' Returns the contents of the file specified by absolute path '''
        with open(path,"rb") as data:
            lattice = pickle.load(data)
        return lattice
        
    def preserve_second_moments(self):
        ''' This method generates a synthetic data set that presevers the second order moments
        of the original data.
        private_data: can be a ContinuousGenerator object that stores the data in the features/targets property or a DataLoader object'''
        
        # Initialise targets and features lattice
        self.intialise_lattices()
        
        # Generate outcome space 
        # TODO: Design OutcomeSpaceGenerator class 
        self.OutcomeSpace.generate_outcomes(self.outcome_features,self.outcome_targets,self.private_data,self.synth_data_type,self.epsilon)
        # TODO: Design Sampler() class
        self.Sampler()
        
    def preserve_first_moments(self):
        ''' This method generates a synthetic data set that presevers the first order moments
        of the original data '''
        pass
    
    def generate_data(self,private_data,num_points=5):
        
        self.private_data = private_data
        self.num_points = num_points
        self.dimensionality = private_data.features.shape[1]
        
        if self.synth_data_type == 'second_moments':
            self.preserve_second_moments()
        elif self.synth_data_type == 'first_moments':
            self.preserve_first_moments()
        else:
            # TODO: Error handling 
            pass
        
        
        



        