# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 18:12:17 2018

@author: alexc
"""
from Generators import ContinuousGenerator

# Exploration into the sensitivity calculation...

# Generate a continuous data set... with row 2-norm bounded by 1

DataGenerator = ContinuousGenerator(d=4,n=20)
DataGenerator.generate_data(bound_recs=True)
features = DataGenerator.features


