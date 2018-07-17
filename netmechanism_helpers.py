# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:18:23 2018

@author: alexc
"""

import pickle, os
import numpy as np

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
                try:
                    data.append(self.load_batch_scores(filenames[entry]))
                except IndexError:
                    print ("Attempted to access filename of index", entry)
        return data
    
    def save_synthetic_data(self, data, directory, filename):
        
        directory = directory.replace(os.path.basename(directory),'SyntheticData') 
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        fullpath = directory + "/" + filename
        
        if os.path.exists(fullpath):
            rand_int = np.random.choice(range(1000000))
            fullpath = fullpath + "_" + str(rand_int)

       
        with open(fullpath,"wb") as container:
            pickle.dump(data, container)
            
    def load_lattice(self,path):
        ''' Returns the contents of the file specified by absolute path '''
        with open(path,"rb") as data:
            lattice = pickle.load(data)
        return lattice
        
    def save_lattice(self, folder_path, lattice_name):
        ''' Saves the lattice to the location specified by @folder_name with
        the name specified by @lattice_name.'''
        
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            
        full_path = folder_path + "/" + lattice_name
       
        # Raise an error if the target file exists
        if os.path.exists(full_path):
            raise FileExistsError("Lattice has already been saved!")
        
        raise NotImplementedError
        
    def save(self, data, path, filename):
        
        if not os.path.exists(path):
            os.makedirs(path)
            
        fullpath =  path + "/" + filename     
        
        if os.path.exists(fullpath):
            raise FileExistsError("Please delete or copy the old data!")
            
        with open(fullpath ,"wb") as container:
            pickle.dump(data, container)
        
    