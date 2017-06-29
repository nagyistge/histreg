#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-07 First release.
# Description: fits a plane and a special type of quadric surface onto given
# points in xyz. Loads MR volume, slices along the surface and projects it
# orthogonally onto the plane. Parallel sclices are gathered while translating
# the surface along the normal vector of the plane. The result is saved into a
# NIfTI file.

# 2017-Jun-29
# Completely redesigned, simplified and rewritten. Quadratic fit is performed
# relative to a grid in the fitted plane. No discriminant errors, easier multi-
# nomial expansion.

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as npp
import nibabel as nib
from scipy.interpolate import RegularGridInterpolator
import tifffile as tiff
from mpl_toolkits.mplot3d import Axes3D


def polyfit2D(x, y, z, order, restrict_termorder=True):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    vander = npp.polyvander2d(x, y, [order]*2)
    vander = vander.reshape((-1, vander.shape[-1]))
    if restrict_termorder:
        termorder = np.sum(
            np.dstack(np.meshgrid(range(order + 1), range(order + 1))), axis=-1)
        coefficients = np.linalg.lstsq(vander[:,
                                       np.flatnonzero((termorder<=order))],
                            z.reshape((-1),))[0]
        all_coefficients = np.zeros(vander.shape[-1])
        all_coefficients[(termorder<=order).ravel()] = coefficients
        return all_coefficients.reshape((order+1, order+1))
    else:
        return np.linalg.lstsq(vander, z.reshape((-1),))[0]\
            .reshape((order+1, order+1))

# Load the MR volume
print 'Loading the MR volume...'
fpath = '/Volumes/INH_1TB/MND_HistReg_Scratch/bbr3d/old/'
fname = 'structural_bet_f01_brain.nii.gz'
vol = nib.load(fpath + fname).get_data()
print vol.shape
nx, ny, nz = vol.shape
mri_ipol = RegularGridInterpolator((range(nx), range(ny), range(nz)), vol,
                                   bounds_error=False, fill_value=0)

# Load fix points
print 'Loading the table of fix points...'
points = np.loadtxt('slice14B_coords.txt') # [[x, y, z]]
sform = np.loadtxt('sform_custom.mat')
points = np.dot(sform[:3, :], np.vstack((points.T, np.ones(points.shape[0])))).T
x, y, z = np.hsplit(points, 3)

# Fit plane
print 'Fitting plane...'
lin_params = polyfit2D(x, z, y, order=1, restrict_termorder=True)
D, C, A, _ = lin_params.reshape(-1)
B = 1

# Calculate the unit normal vector for the plane
snorm = np.array([[A], [B], [C]])
snorm = snorm / np.linalg.norm(snorm) # this is really important

# Define MR space base
i0 = np.array([[1], [0], [0]])
j0 = np.array([[0], [1], [0]])
k0 = np.array([[0], [0], [1]])
B0 = np.hstack((i0, j0, k0))

# Define new base
j1 = np.copy(snorm) # surface normal as second base vector
k1 = k0 - np.dot(k0.T, snorm) * snorm
i1 = np.cross(j1, k1, axisa=0, axisb=0).T
i1 = i1 / np.linalg.norm(i1)
j1 = j1 / np.linalg.norm(j1)
k1 = k1 / np.linalg.norm(k1)
B1 = np.hstack((i1, j1, k1))

# Calculate the translation vector
T = -D * snorm
T = np.zeros((3,1))

# Fit quadratic surface for in-plane coordinate pairs
print 'Fitting quadratic surface...'
px, py, pz = np.vsplit(np.dot(np.dot(np.linalg.inv(B1), B0), points.T) +
                       np.dot(np.linalg.inv(B1), T), 3)

quad_params = polyfit2D(px, pz, py, order=1, restrict_termorder=False)
step = 0.234375
x_min, y_min, z_min = np.vsplit(np.dot(sform[:3, :],
                                       np.array([[nx-1], [0], [0], [1]])), 3)
x_max, y_max, z_max = np.vsplit(np.dot(sform[:3, :],
                                       np.array([[0], [ny-1], [nz-1], [1]])), 3)
npx = np.arange(x_min, x_max, step)
npz = np.arange(z_min, z_max, step)
gz, gx = np.meshgrid(npz, npx, indexing='ij')
gy = npp.polyval2d(gx, gz, quad_params)
sampling_coordinates = np.dot(np.dot(np.linalg.inv(B0), B1),
                       np.vstack((gx.ravel(), gy.ravel(), gz.ravel()))) - \
                       np.dot(np.linalg.inv(B0), T)

sampling_voxel_coordinates = np.dot(np.linalg.inv(sform)[:3, :],
                                    np.vstack((sampling_coordinates,
                                               np.ones_like(gx.ravel())))).T

# Sanity check: plot the surfaces
pts = 50
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot3D(x.ravel(), y.ravel(), z.ravel(), "rx")
ax.plot3D([0], [0], [0], "bo")
ax.plot_surface(gx, npp.polyval2d(gx, gz, lin_params), gz, color='green')
pgx, pgy, pgz = np.vsplit(np.dot(np.dot(np.linalg.inv(B1), B0),
                                 np.vstack((gx.ravel(), gy.ravel(), gz.ravel()))) + \
                                 np.dot(np.linalg.inv(B1), T), 3)
ax.plot_surface(pgx.reshape(gx.shape), npp.polyval2d(pgx.reshape(gx.shape),
                                                     pgz.reshape(gz.shape),
                                                     quad_params),
                pgz.reshape(gz.shape), color='cyan')
lo, hi = np.min(points), np.max(points)
ax.set_xlim([lo, hi])
ax.set_ylim([lo, hi])
ax.set_zlim([lo, hi])
fig.canvas.draw()
plt.show()

# Sample values from the MRI volume (actual reslicing)
intensity_values = mri_ipol(sampling_voxel_coordinates).reshape((gx.shape))
#plt.imshow(intensity_values[:,::-1].T, cmap='gray')
#plt.show()

# Create stack from multiple parallel reslices
stack = []
slicerange = np.arange(-10, 10, step)
slicerange = np.array([0])
for r in slicerange:
    print 'Sampling @ {} mm'.format(r)
    sampling_coordinates = np.dot(np.dot(np.linalg.inv(B0), B1),
                                  np.vstack(
                                      (gx.ravel(), gy.ravel()+r, gz.ravel()))) - \
                           np.dot(np.linalg.inv(B0), T)
    sampling_voxel_coordinates = np.dot(np.linalg.inv(sform)[:3, :],
                                        np.vstack((sampling_coordinates,
                                                   np.ones_like(gx.ravel())))).T
    intensity_values = mri_ipol(sampling_voxel_coordinates).reshape((gx.shape))
    stack.append(intensity_values[:,::-1].T) # set orientation

# Save the slices
niftifile = '../../newreslicer/mri_quad_slice'
stack = np.dstack(stack)
print stack.shape
sform = np.loadtxt('../../bbr3d/sform_iso.mat')
sform[2,3] = sform[2,3] + slicerange[0]
hdr = nib.Nifti1Header()
hdr.set_data_shape((npx.size, npz.size, slicerange.size))
hdr.set_dim_info(slice=2)
hdr.set_xyzt_units(2, 16)  # nifti codes for mm and msec
hdr.set_qform(sform, 1, strip_shears=True)
hdr.set_sform(sform, 2)
nimg = nib.Nifti1Image(stack, sform, hdr)
nib.save(nimg, niftifile + '.nii.gz')
print 'NIfTI image was created at {}.'.format(niftifile + '.nii.gz')

# Save as tiff
stack = np.swapaxes(stack, 0, 1)
tiff.imsave(niftifile + '.tif', stack[::-1,:,:].astype(np.float32))
print 'TIFF image was created at {}.'.format(niftifile + '.tif')