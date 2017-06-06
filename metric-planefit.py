#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# This code was downloaded from:
# https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load voxel coordinates (i,j,k) of known points
data = np.loadtxt('slice14B_coords.txt')
# Load affine matrix
sform = np.loadtxt('sform_custom.mat')
# Calculate metric positions in "standard" space
data = np.dot(np.hstack((data, np.ones((data.shape[0], 1)))), sform[:3,:].T)
print data
#np.savetxt('slice_14B_coords_mm.txt', data)

print 'Min (x,y,z):', np.min(data,axis=0)
print 'Max (x,y,z):', np.max(data, axis=0)

# regular grid covering the domain of the data
n_points = 100
X, Y = np.meshgrid(np.linspace(np.min(data[:,0]), np.max(data[:,0]), n_points), np.linspace(np.min(data[:,1]), np.max(data[:,1]), n_points))

# best-fit linear plane
A = np.c_[data[:, 0], data[:, 1], np.ones((data.shape[0], 1))]
C, residues, _, _ = scipy.linalg.lstsq(A, -data[:, 2])  # coefficients

# evaluate it on grid
Z = -(C[0] * X + C[1] * Y + C[2])
slicecoords = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
#np.savetxt('slicecoords.txt', slicecoords)

print 'Coefficients:', C
print 'Residues:', residues

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d((-55, 35))
ax.set_ylim3d((-55, 35))
ax.set_zlim3d((-55, 35))
#ax.axis('equal')
#ax.axis('tight')
plt.show()