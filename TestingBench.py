# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:24:09 2018

@author: alexc
"""

#Useful plotting indicatiors r--,bs,g^,b.,b*

from Generators import DataGenerator
import matplotlib.pyplot as plt
import numpy as np
import math

#%% Test regression line using DataGenerator object
#data = generators.DataGenerator(reg_slope=1,reg_intercept=0,num_pts_y_lattice=10)
#line = data.generate_reg_line()
#plt.plot(line[:,0],line[:,1],'g^')
#%% Testing lattice generation
#data = generators.DataGenerator(reg_slope=2,reg_intercept=1,num_pts_y_lattice=10)
#lattice = data.generate_lattice()
#points = data.plot_lattice(lattice)
#%% Test that generate_data_set correctly sets the properties lattice and regression_line and x_coordinate generation
#generator = generators.DataGenerator()
#test = generator.lattice 
#print(test)
#DataSet = generator.generate_data_set(num_pts=15,reg_slope=2,reg_intercept=1,num_pts_x_lattice=100,num_pts_y_lattice=50)
#lattice = generator.lattice
#regression_line = generator.regression_line
#x_coordinates = generator.x_locs
#print(lattice)
#print(regression_line)
#%% Test generation on that when more outputs are required compared to the number of lattice x coordinates
generator = DataGenerator(batch_size=15)
generator.generate_data_set(num_pts=60,reg_slope=2,reg_intercept=1,num_pts_x_lattice = 20,num_pts_y_lattice = 50)
data_set = generator.data
plt.figure(0)
plt.scatter(data_set[:,0],data_set[:,1])
generator = DataGenerator()
generator.generate_data_set(num_pts=35,reg_slope=2,reg_intercept=1,num_pts_x_lattice = 100,num_pts_y_lattice = 50)
data_set = generator.data
plt.figure(1)
plt.scatter(data_set[:,0],data_set[:,1])
generator = DataGenerator()
generator.generate_data_set(num_pts=15,reg_slope=3,reg_intercept=0.5,num_pts_x_lattice = 50,num_pts_y_lattice = 50)
data_set = generator.data
plt.figure(2)
plt.scatter(data_set[:,0],data_set[:,1])


