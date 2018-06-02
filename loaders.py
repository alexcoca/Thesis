# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:04:16 2018

@author: alexc
"""
import numpy as np

class DataLoader():
    
    def __init__(self):
        self.raw_data = []
        self.features = []
        self.targets = []
        self.data = []
        self.categorical = False
        self.target_names = []
        self.feature_names = []
        
    def load_raw_data (self,path):
        ''' This method reads the data from the raw file and performs the
        pre-processing steps listed in the @process function'''
        def process(line):
            ''' Helper function: converts entries to floats.
            & remove categorical features for  each data point (aka line)'''
            processed_line = []
            for entry in line.split():
                if entry.isalpha():
                    processed_line.append(entry)
                else:
                    processed_line.append(float(entry))
            processed_line=[processed_line[i] for i in self.features+self.targets]
            return processed_line
        
        with open(path) as f:
            first_line = f.readline() 
            self.raw_data.append(process(first_line))
            first_line = first_line.split()
            for col_index in self.features:   # Check if there are categorical features
                if first_line[col_index].isalpha() == True:
                    self.categorical = True
                    break
            for line in f:
                self.raw_data.append(process(line))
        if self.categorical:
            # Create feature and target names
            feature_names = ['feature'+str(i) for i in range(len(self.features))]
            target_names =  ['target'+str(i) for i in range(len(self.targets))]
            names = feature_names + target_names
            # Structure data as a record array
            self.raw_data = np.rec.fromrecords(self.raw_data,names=names)
            self.feature_names = feature_names
            self.target_names = target_names
        else: 
            self.raw_data = np.array(self.raw_data,dtype=float)

    
    def load(self,path,features=-1,targets=-1,num_samples=-1,max_feat_idx=10,max_targ_idx=13):
        '''
        This function loads the data from the file specified by path
        and selects the features and targets specified in the corresponding vectors. 
        If @num_samples is specified (i.e., !=-1) then num_samples records are sampled from the 
        dataset.
        If @features is set to -1 then all the features (columns 1-10 in the original data set) are selected
        If @targets is set to -1 all targets (columns 11-13 in the original data set) are selected.
        Other parameters:
            max_feat_idx: number of features the data set contains. Features are assumed to be in consecutive columns
            max_targ_idx: number of targets the data set contains. Targets are assumed to be in consecutive columns starting
            after the feature columns.
        '''
        if features == -1:
            features = list(range(max_feat_idx))            
        if targets == -1:
            targets = list(range(max_feat_idx,max_targ_idx))
    
        # Get raw data
        self.features = features
        self.targets = targets
        self.load_raw_data(path)
        
        # Return all data samples
        if num_samples == -1:
            if not self.categorical:
                features_idx = np.arange(len(features))
                self.features = self.raw_data[:,features_idx]
                targets_idx = features_idx[-1]+1+np.arange(len(targets))
                self.targets = self.raw_data[:,targets_idx]
                self.data =np.concatenate((self.features,self.targets),axis=1)
            else:
                self.features = self.raw_data[self.feature_names]
                self.targets = self.raw_data[self.target_names]
                