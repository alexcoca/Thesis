# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:52:29 2018

@author: alexc
"""
#%%
# Plot a lattice inside a sphere
from itertools import product, combinations
from Generators import ContinuousGenerator 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# define axes

fig = plt.figure(figsize=(20,10))
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r, r, r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s, e), color="b")

# Create a sphere
r = 1.0
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)
ax.plot_wireframe(x,y,z,alpha=0.2,rstride=5,cstride=5)

# draw points
#path = 'C:/Users/alexc/OneDrive/Documents/GitHub/Thesis/Data/plots/poster/pts_inside_sphere.pickle'
#with open(path,'rb') as src_data:
#    points = pickle.load(src_data)
ax.scatter(points[:,0], points[:,1], points[:,2], color="g", s=50)
#plt.rcParams.update('figure.figsize',[8,6])
plt.show()

print(plt.rcParams) # to examine all values
 
print(plt.rcParams.get('figure.figsize'))