#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-14
# My attempt for illumination-independent rigid body registration of
# histological blocks and brain slices.

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos, pi, floor

fpath = '../../block_insertion/'

# Load images
slice = tiff.imread(fpath+'slice_seg_rot.tif')[:,:,:3]
fragment = tiff.imread(fpath+'fragment_seg.tif')[:,:,:3]
block = tiff.imread(fpath+'block_seg.tif')[:,:,:3]

# Downsample images
def downsample(img, factor, bounds_error=False, fill_value=0):
    h, w, ch = img.shape
    nx = np.linspace(0, w, floor(w / factor) + 1, False)
    ny = np.linspace(0, h, floor(h / factor) + 1, False)
    nyy, nxx = np.meshgrid(ny, nx, indexing='ij')
    newimg = np.zeros((ny.size, nx.size, ch), dtype=np.uint8)
    for c in range(ch):
        ipol = RegularGridInterpolator((range(h), range(w)), img[:, :, c],
                                       bounds_error=bounds_error,
                                       fill_value=fill_value)
        newimg[:, :, c] = ipol(np.vstack((nyy.ravel(), nxx.ravel())).T)\
                          .reshape(nxx.shape).astype(np.uint8)
    return newimg

ppmm = 16.83495/2.0
slice = downsample(slice, factor=ppmm)
fragment = downsample(fragment, factor=ppmm)
block = downsample(block, factor=ppmm)

# Re-obtain shapes
sh, sw, _ = slice.shape
fh, fw, _ = fragment.shape
bh, bw, _ = block.shape

# Display images
"""
plt.subplot(131)
plt.imshow(slice)
plt.subplot(132)
plt.imshow(fragment)
plt.subplot(133)
plt.imshow(block)
plt.show()
"""

# Define image interpolators for the slice and the fragment
slice_ipol = RegularGridInterpolator((range(sh), range(sw)), slice,
                                     bounds_error=False, fill_value=np.array([0,0,0]))
fragment_ipol = RegularGridInterpolator((range(fh), range(fw)), fragment,
                                        bounds_error=False, fill_value=np.array([0,0,0]))

def transform_img(img, ipol, affine_params):
    testimg = ipol(
              transform(img, affine_params)).reshape(img.shape).astype(np.uint8)
    return testimg

# Define rigid-body transformer function
# See https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
def transform(img, affine_params):
    """ Returns new x and y coordinates """
    h, w, ch = img.shape
    #ext_range = range(w) + range(h)
    rot_z, dx, dy, sx, sy = affine_params
    centre_x, centre_y = np.mean(np.vstack(np.where(np.sum(img, axis=2)!=0)).T,
                                 axis=0)
    dx = dx - centre_x
    dy = dy - centre_y
    tx = -(-centre_x) * sx * cos(rot_z) + (-centre_y) * sy * sin(rot_z)
    ty = -(centre_x) * sx * sin(rot_z) - (centre_y) * sy * cos(rot_z)
    affine = np.array([[sx*cos(rot_z), -sy*sin(rot_z),
                        dx*sx*cos(rot_z)-dy*sy*sin(rot_z)+tx],
                       [sx * sin(rot_z), sy * cos(rot_z),
                        dx * sx * sin(rot_z) - dy * sy * cos(rot_z)+ty],
                       [0, 0, 1]])

    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
    res = np.dot(affine, np.vstack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
    res = np.roll(res[:2, :].T, 1, axis=1)
    return res

# Define cost function
def costfun(paramvect, slice_ipol, fragment_ipol, slice):
    xy_slice = transform(slice, paramvect)
    #ext_range = np.linspace(0, sh+sw, sh+sw, False) #+ (sh+sw)/2
    yy, xx = np.meshgrid(range(sh), range(sw), indexing='ij')
    xy_fragment = np.vstack((yy.ravel(), xx.ravel())).T
    cost = np.sum((fragment_ipol(xy_fragment).reshape(slice.shape) -
                   slice_ipol(xy_slice).reshape(slice.shape))**2)
    print cost
    """
    plt.subplot(131)
    plt.imshow(fragment_ipol(xy_fragment).reshape(slice.shape), cmap='gray')
    plt.subplot(132)
    plt.imshow(slice_ipol(xy_slice).reshape(slice.shape),cmap='gray')
    plt.subplot(133)
    plt.imshow(np.sum((fragment_ipol(xy_fragment).reshape(slice.shape) -
                   slice_ipol(xy_slice).reshape(slice.shape))**2, axis=2), cmap='gray')
    plt.show()
    """
    return cost

# Register the slice and the fragment
print 'Optimizing...'
bnds = ((-pi/2.0, pi/2.0), (-100, 100), (-100, 100), (0.8, 1.2), (0.8, 1.2))
opt = minimize(costfun, np.array([0, 0, 0, 1, 1]),
               args=(slice_ipol, fragment_ipol, slice), bounds=bnds)
print opt.x

# Test results
testimg = np.zeros_like(slice, dtype=np.uint8)
testimg = slice_ipol(transform(slice, opt.x)).reshape(slice.shape)\
          .astype(np.uint8)
plt.subplot(121)
plt.imshow(testimg)
plt.subplot(122)
plt.imshow(fragment)
plt.show()

h = min(testimg.shape[0], fragment.shape[0])
w = min(testimg.shape[1], fragment.shape[1])
plt.imshow(testimg[:h,:w,:]-fragment[:h,:w,:])
plt.show()