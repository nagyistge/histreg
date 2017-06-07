#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-07 First release.
# Description: fits a plane and a special type of quadric surface onto given
# points in xyz. Loads MR volume, slices along the surface and projects it
# orthogonally onto the plane. Parallel sclices are gathered while translating
# the surface along the normal vector of the plane. The result is saved into a
#  NIfTI file.

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
from itertools import product
import tifffile as tiff
from mpl_toolkits.mplot3d import Axes3D


# Matrix of permuted variable products for the terms of the polynomial sum
def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        if i+j <= order:  # I added this line
            G[:, k] = x**i * y**j
    return G

# Load fix points
print 'Loading the table of fix points...'
points = np.loadtxt('slice14B_coords.txt')
sform = np.loadtxt('sform_custom.mat')
points = np.dot(np.hstack((points, np.ones((points.shape[0], 1)))), sform[:3,:].T)

# Fit plane (Ax + By + z + D = 0) to the points
print 'Fitting plane...'
x, y, z = points.T

G = poly_matrix(x, y, 1)
D, B, A, _ = np.linalg.lstsq(G, -z)[0]
print 'Plane coefficients: A={}, B={}, C=1, D={}'.format(A,B,D)
#a, b, d = (0.32591914, -5.85479878, -165.87041371)
snorm = np.array([[A], [B], [1]])
snorm = snorm / np.linalg.norm(snorm) # this is really important

# Fit quadratic surface (ax + by + cxx + dyy + exy + z + f = 0) to the points
print 'Fitting quadratic surface...'
G = poly_matrix(x, y, 2)
a, b, c, d, e, _, f, _, _ = np.linalg.lstsq(G, -z)[0]
print np.linalg.lstsq(G, -z)[0]

# Define base metric in the MR space
i0 = np.array([[1], [0], [0]])
j0 = np.array([[0], [1], [0]])
k0 = np.array([[0], [0], [1]])
M0 = np.hstack((i0, j0, k0))

# Define new 3-base metric
j1 = np.copy(snorm) # surface normal as second base vector
k1 = k0 - np.dot(k0.T, snorm) * snorm
i1 = np.cross(j1, k1, axisa=0, axisb=0).T

# Normalise new base vectors
i1 = i1 / np.linalg.norm(i1)
j1 = j1 / np.linalg.norm(j1)
k1 = k1 / np.linalg.norm(k1)
M1 = np.hstack((i1, j1, k1))

# Load the MR volume

print 'Loading the MR volume...'
#fpath = ''
#fname = 'tagged_volume_quadratic.nii.gz'
fpath = '/Volumes/INH_1TB/MND_HistReg_Scratch/bbr3d/old/'
fname = 'structural_bet_f01_brain.nii.gz'

vol = nib.load(fpath+fname).get_data()
print vol.shape
nx, ny, nz = vol.shape

# Load sform affine matrix
print 'Loading the affine matrix...'
sform = np.loadtxt('sform_custom.mat')

# Calculate rotation matrix
print 'Calculating the rotation matrix...'
R = np.dot(M1, np.linalg.inv(M0))
print R

# Calculate translation vector: t = C' - RC
# Simplifying by setting C = [0,0,0].T: t = C'
# Where C' is the orthogonal projection of the origin to the plane
print 'Calculating the translation vector...'
raw_norm = np.array([[A], [B], [1]])
t = -D * raw_norm / np.linalg.norm(raw_norm) ** 2
print t

# Display full affine matrix
print 'Full affine matrix (saved as reslice_affine.mat):'
affine = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
print affine
np.savetxt('reslice_affine.mat', affine, fmt='%-1.6f')
affine_inv = np.linalg.inv(affine)
np.savetxt('reslice_affine_inv.mat', affine_inv, fmt='%-1.6f')
print '(Full affine matrix inverse was saved as reslice_affine_inv.mat)'


# Transform coordinates from voxels to metric
print 'Transforming volume coordinates into metric space...'
XXX, YYY, ZZZ = np.meshgrid(np.linspace(0,nx,nx,False), np.linspace(0,ny,ny,False), np.linspace(0,nz,nz,False), indexing='ij')
coords = np.vstack((XXX.reshape((1, -1), order='C'), YYY.reshape((1, -1), order='C'), ZZZ.reshape((1, -1), order='C'), np.ones((1, nx*ny*nz))))
coords = np.dot(sform[:3,:], coords)

X_mm, Y_mm, Z_mm = tuple(np.split(coords.reshape((3,nx,ny,nz), order='C'), 3, axis=0))
X_mm = X_mm[0,:,0,0]
Y_mm = Y_mm[0,0,:,0]
Z_mm = Z_mm[0,0,0,:]

# Set up interpolator
# Since sform imparts a linear transformation to a regular voxel grid,
# the RegularGridInterpolator can be used on the resulting metric grid.
# According to its documentation, the grid may have uneven spacing.
print 'Setting up the interpolator...'
if X_mm[-1] <= X_mm[0]:
    X_mm = X_mm[::-1]
    vol = vol[::-1,:,:]
if Y_mm[-1] <= Y_mm[0]:
    Y_mm = Y_mm[::-1]
    vol = vol[:,::-1,:]
if Z_mm[-1] <= Z_mm[0]:
    Z_mm = Z_mm[::-1]
    vol = vol[:,:,::-1]
ipol = RegularGridInterpolator((X_mm, Y_mm, Z_mm), vol, bounds_error=False, fill_value=0)
print 'Interpolator setup complete.'

# Determine bounds from the original volume edge intersections with the plane
# I will add this later

# Create a meshgrid from two new bases:
print 'Reslicing...'
npx = int((np.max(X_mm) - np.min(X_mm))/0.234375 + 1)
npz = int((np.max(Z_mm) - np.min(Z_mm))/0.234375 + 1)
Xs = np.linspace(np.min(X_mm), np.max(X_mm), npx)
Zs = np.linspace(np.min(Z_mm), np.max(Z_mm), npz)
XX_plane, ZZ_plane = np.meshgrid(Xs, Zs, indexing='ij')

# Obtain xyz coordinates of the (projection) points in the meshgrid
planegrid = np.dstack((XX_plane, np.zeros_like(XX_plane), ZZ_plane,
                       np.ones_like(XX_plane))).reshape((-1, 4),
                                                        order='C').T
X0, Y0, Z0 = np.dot(affine[:3, :], planegrid)
planegrid = np.vstack((X0, Y0, Z0)).T
#print 'Planegrid:', planegrid.shape

# Check plane
"""
planeslice = ipol(planegrid)
planeslice = planeslice.reshape(npx,npz)
plt.imshow(planeslice[:,::-1].T, cmap='gray')
plt.show()
"""

# Calculate projection parameter for all (projection) points in the meshgrid
alpha = np.repeat(c * B ** 2 + e * A * B + f * A ** 2, npx * npz, axis=0)
beta = X0 * (e * B + 2 * f * A) + Y0 * (2 * c * B + e * A) + b * B + d * A + 1
gamma = d * X0 + b * Y0 + f * X0 ** 2 + e * X0 * Y0 + c * Y0 ** 2 + a + Z0
discr = np.sqrt(beta ** 2 - 4 * alpha * gamma)
# print 'Unique discriminants:', np.unique(discr)
tau = np.vstack(((-beta - discr) / (2 * alpha), (-beta + discr) / (2 * alpha)))
# print 'Tau:', tau.shape
X = np.repeat(X0[np.newaxis, :], 2, axis=0) + A * tau
Y = np.repeat(Y0[np.newaxis, :], 2, axis=0) + B * tau
Z = np.repeat(Z0[np.newaxis, :], 2, axis=0) + tau
intersections = np.dstack((X, Y, Z))
# print 'Intersections:', intersections.shape
# Chose the closer one from the intersections
dist = np.linalg.norm(
    intersections - np.repeat(planegrid[np.newaxis, :, :], 2, axis=0), axis=2)
# print 'Dist:', dist.shape
idx = np.argmin(dist, axis=0)
projections = []
for i in xrange(idx.size):
    projections.append(intersections[idx[i], i, :])
projections = np.array(projections)

# Check integrity
"""
X = projections[:,0]
Y = projections[:, 0]
Z = projections[:, 0]
checksum = a + b*Y + c*Y**2 + d*X + e*X*Y + f*X**2 + Z
print 'Checksum:', checksum[checksum!=0].size
checknorm = projections - np.vstack((X0, Y0, Z0)).T
for i in xrange(checknorm.shape[0]):
    print checknorm[i,:]/checknorm[i,-1]

# Plot surface in xyz space
fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot3D(projections[:,0], projections[:,1], projections[:,2], "o")
fg.canvas.draw()
plt.show()
"""

stack = []
for r in 0.234375*np.linspace(-25, 25, 51):
    print 'Slicing @ {} mm...'.format(r)
    # Acquire the image of the orthogonally projected quadratic slice
    flatimg = ipol(projections + r * M1[:,1].T)
    flatimg = flatimg.reshape((npx, npz))

    # Add to the stack
    stack.append(flatimg)

# Save the slices
stack = np.dstack(stack)
sform = np.loadtxt('../../bbr3d/sform_iso.mat')
#sform[2, 3] = -161.0125
hdr = nib.Nifti1Header()
hdr.set_data_shape((nx, nz, 51))
hdr.set_xyzt_units(2, 16)  # nifti codes for mm and msec
hdr.set_qform(sform, 1, strip_shears=True)
hdr.set_sform(sform, 2)
nimg = nib.Nifti1Image(stack, sform, hdr)
nib.save(nimg, '../../bbr3d/mri_ortho_quadratic_iso_51slices.nii.gz')
print 'NIfTI image was created.'