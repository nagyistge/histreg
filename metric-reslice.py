#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-May-31 This will work. I know. We proved it. :)

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator

print 'Initialising...'

# Define base metric in the MR space
i0 = np.array([[1], [0], [0]])
j0 = np.array([[0], [1], [0]])
k0 = np.array([[0], [0], [1]])
M0 = np.hstack((i0, j0, k0))

# Define plane equation: ax + by + z + d = 0
a, b, d = (0.32591914, -5.85479878, -165.87041371)
snorm = np.array([[a], [b], [1]])
snorm = snorm / np.linalg.norm(snorm) # this is really important

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
#fname = 'tagged_volume.nii.gz'
fpath = '/Volumes/INH_1TB/MND_HistReg_Scratch/bbr3d/'
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
raw_norm = np.array([[a],[b],[1]])
t = -d * raw_norm / np.linalg.norm(raw_norm) ** 2
print t

# Show full affine matrix
print 'Full affine matrix (saved as reslice_affine.mat):'
affine = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))
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
f = RegularGridInterpolator((X_mm, Y_mm, Z_mm), vol, bounds_error=False, fill_value=0)
print 'Interpolator setup complete.'

# Determine bounds from the original volume edge intersections with the plane
# I will add this later

# Create a meshgrid from two new bases:
print 'Reslicing...'
npx = int((np.max(X_mm) - np.min(X_mm))/0.234375 + 1)
npz = int((np.max(Z_mm) - np.min(Z_mm))/0.234375 + 1)
X1 = np.linspace(np.min(X_mm), np.max(X_mm), npx)
Z1 = np.linspace(np.min(Z_mm), np.max(Z_mm), npz)
PX, PZ = np.meshgrid(X1, Z1, indexing='ij')

stack = []
for r in 3*0.234375*np.linspace(-6, 6, 13):
    P = np.vstack((PX.reshape(1,-1), np.zeros_like(PX.reshape(1,-1)) + r, PZ.reshape(1,-1), np.ones_like(PX.reshape(1,-1))))
    #P = np.dstack((PX, np.zeros_like(PX)+r, PZ)).reshape((-1, 3), order='C').T

    # Transform planar meshgrid coordinates into original 3-space coordinates
    V = np.dot(affine[:3,:], P)
    #print V

    # Draw intensity values from the volume by interpolation
    I = f(V.T)

    # Show the slice
    reslice = I.reshape((npx, npz), order='C')
    print 'Reslicing complete.'
    #plt.imshow(reslice[::-1,:], cmap='gray', aspect='equal')
    #plt.show()
    stack.append(reslice[:,:].astype(np.float32))

dummy = np.zeros((npx,npz,50))
stack = np.dstack(stack)
print stack.shape
stack = np.dstack((dummy, stack, dummy))
print stack.shape

sform = np.loadtxt('../../bbr3d/sform_3layer.mat')
sform[2, 3] = -161.0125
hdr = nib.Nifti1Header()
#hdr.set_dim_info(slice=1)  # Set Z as slice encoding axis.
hdr.set_data_shape((nx,nz,113))
hdr.set_xyzt_units(2, 16)  # nifti codes for mm and msec
hdr.set_qform(sform, 1, strip_shears=True)
hdr.set_sform(sform, 2)
nimg = nib.Nifti1Image(stack, sform, hdr)
nib.save(nimg, '../../bbr3d/mri_ortho_3layer_13slices_ext.nii.gz')
print 'NIfTI image was created.'
#tiff.imsave('../../bbr3d/mri_ortho_3layer_13slices_ext.tif', stack)
#print 'Multi-layer TIFF file was saved successfully.'