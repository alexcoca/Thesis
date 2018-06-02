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
            processed_line=processed_line[3:] # Discard categorical features
            return processed_line
        
        raw_data = []
        with open(path) as f:
            for line in f:
                raw_data.append(process(line))
        raw_data = np.array(raw_data,dtype=float)
        return raw_data
    
    def load(self,path,features=-1,targets=-1,num_samples=-1):
        '''This function loads the data from the file specified by path
        and selects the features and targets specified in the corresponding vectors. 
        If @num_samples is specified (i.e., !=-1) then num_samples records are sampled from the 
        dataset.
        If @features is set to -1 then all the features (columns 4-10 in the original data set) are selected
        Note: Categorical attributes (columns 0,1,2 of the original data set) are excluded from 
        the data set so feature 0 corresponds to the 4th column of the original data set
        If @targets is set to -1 all targets (columns 11-13 in original data set) are selected.'''
        
        # Get raw data
        self.raw_data = self.load_raw_data(path)
        
        # Return all data samples
        if num_samples == -1:
            self.features = self.raw_data[:,features]
            self.targets = self.raw_data[:,targets]
                