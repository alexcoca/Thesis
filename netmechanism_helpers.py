# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:18:23 2018

@author: alexc
"""

import pickle

class FileManager(object):
    
    def __init__(self):
        pass
    
    def load_batch_scores(self, path):
        ''' Returns the contents of the file specified by absolute path '''
        with open(path, "rb") as data:
            batch_scores = pickle.load(data)
        return batch_scores
    
    def retrieve_scores(self,filenames,batches=[]):
        """ This method unpickles the files listed in the @filenames list, 
        returning a list containing the contents of the unpickled files.
        If @batches list is specified, then only the files to the corresponding
        to entries of the list are loaded """
    
        def get_batch_id(filename):
            return int(filename[filename.rfind("_") + 1:])
            
        data = []
        
        # Filenames have to be sorted to ensure correct batch is extracted
        filenames  = sorted(filenames, key = get_batch_id)
        
        if not batches:        
            for filename in filenames:
                data.append(self.load_batch_scores(filename))
        else:
            for entry in batches:
                data.append(self.load_batch_scores(filenames[entry]))
        return data