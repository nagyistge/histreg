#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-14
# My attempt for illumination-independent rigid body registration of
# histological blocks and brain slices.

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.optimize import minimize, basinhopping
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos, pi, floor, sqrt
from fractions import Fraction

fpath = '../../block_insertion/'

# Load images
slice = tiff.imread(fpath+'slice_seg_rot3.tif')[:,:,:3]
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

ppmm = 16.83495/3.0
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

# Calculate local colour gradients
def colour_grad(img, kernelsize=3):
    h, w, ch = img.shape
    harmonic = lambda n: sum(Fraction(1, d) for d in range(1, n + 1))
    U = np.ones((1, kernelsize))
    D = 2.0 / (kernelsize * (kernelsize - 1)) * \
        np.array([float(harmonic(i) - harmonic(kernelsize - 1 - i))
                  for i in range(kernelsize)])
    offset = kernelsize / 2  # integer division is assumed
    dx = np.zeros_like(img, dtype=np.float64)
    dy = np.zeros_like(img, dtype=np.float64)
    for c in range(ch):
        for y in xrange(offset, h - (kernelsize - offset)):
            #print '{}/{} row...'.format(y - offset + 1, h - kernelsize)
            for x in xrange(offset, w - (kernelsize - offset)):
                K = img[y - offset:y - offset + kernelsize,
                        x - offset:x - offset + kernelsize, c]
                current_dx = np.dot(np.dot(U, K), D.T)
                current_dy = np.dot(D, np.dot(K, U.T))
                magnitude = sqrt(current_dx ** 2 + current_dy ** 2)
                if magnitude == 0:
                    dx[y, x, c] = 0
                    dy[y, x, c] = 0
                    continue
                dx[y, x, c] = current_dx * 1.0 / magnitude
                dy[y, x, c] = current_dy * 1.0 / magnitude
    return np.sqrt(dx**2 + dy**2)

# Define image interpolators for the slice and the fragment
slice_ipol = RegularGridInterpolator((range(sh), range(sw)), slice,
                                     bounds_error=False, fill_value=np.array([0,0,0]))
print 'Calculating colour gradient map for the slice...'
slice_grad_ipol = RegularGridInterpolator((range(sh), range(sw)), colour_grad(slice, 10),
                                     bounds_error=False, fill_value=np.array([0,0,0]))
fragment_ipol = RegularGridInterpolator((range(fh), range(fw)), fragment,
                                        bounds_error=False, fill_value=np.array([0,0,0]))
print 'Calculating colour gradient map for the fragment...'
fragment_grad_ipol = RegularGridInterpolator((range(fh), range(fw)), colour_grad(fragment, 10),
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
    rot_z, dx, dy, sx = affine_params
    sy = sx
    centre_y, centre_x = np.mean(np.vstack(np.where(np.sum(img, axis=2)!=0)).T,
                                 axis=0)
    dx = dx - centre_x
    dy = dy - centre_y
    affine = np.array([[sx*cos(rot_z), -sy*sin(rot_z),
                        dx*sx*cos(rot_z)-dy*sy*sin(rot_z) + centre_x],
                       [sx * sin(rot_z), sy * cos(rot_z),
                        dx * sx * sin(rot_z) + dy * sy * cos(rot_z) + centre_y],
                       [0, 0, 1]])
    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
    res = np.dot(np.linalg.inv(affine), np.vstack((xx.ravel(), yy.ravel(), np.ones(xx.size))))
    res = np.roll(res[:2, :].T, 1, axis=1)
    return res

# Test transformer function
"""
plt.imshow(transform_img(slice, slice_ipol, np.array([-pi,0,0,1,1])))
plt.show()
"""

# Define cost function
def costfun(paramvect, slice_ipol, fragment_ipol, sg_ipol, fg_ipol, slice,
            verbose=False):
    alpha = 1
    xy_slice = transform(slice, paramvect)
    yy, xx = np.meshgrid(range(sh), range(sw), indexing='ij')
    xy_fragment = np.vstack((yy.ravel(), xx.ravel())).T
    lsqterm = np.sum((fragment_ipol(xy_fragment).reshape(slice.shape) -
                   slice_ipol(xy_slice).reshape(slice.shape))**2)
    regterm = np.sum((fg_ipol(xy_fragment).reshape(slice.shape) -
                   sg_ipol(xy_slice).reshape(slice.shape))**2)
    cost = lsqterm + alpha*regterm
    if verbose:
        print cost
    """
    plt.subplot(131)
    plt.imshow(fragment_ipol(xy_fragment).reshape(slice.shape), cmap='gray')
    plt.subplot(132)
    print xy_slice
    plt.imshow(slice_ipol(xy_slice).reshape(slice.shape),cmap='gray')
    plt.subplot(133)
    plt.imshow(np.sum((fragment_ipol(xy_fragment).reshape(slice.shape) -
                   slice_ipol(xy_slice).reshape(slice.shape))**2, axis=2), cmap='gray')
    plt.show()
    """
    return cost

# Register the slice and the fragment
bnds = ((-pi, pi), (-100, 100), (-100, 100), (0.8, 1.2))
gridpts = tuple([np.linspace(bnds[i][0], bnds[i][1], 5, True) for i in range(4)])
grid = np.meshgrid(*gridpts)
grid = np.vstack([grid[i].ravel() for i in range(len(grid))]).T
#print np.any([np.all(grid[i,:] == np.array([-pi,0,0,1])) for i in range(grid.shape[0])])
initial_costs = []
print 'Calculating initial guess...'
for x0 in grid:
    initial_costs.append(costfun(x0, slice_ipol, fragment_ipol, slice_grad_ipol,
                                 fragment_grad_ipol, slice))
x0 = grid[np.argmin(np.vstack(initial_costs))]
print x0
print 'Optimizing...'
opt = minimize(costfun, x0=x0, bounds=bnds,
                   args=(slice_ipol, fragment_ipol, slice_grad_ipol,
                     fragment_grad_ipol, slice, True))
"""
opt = basinhopping(costfun, x0=np.array([0, 0, 0, 1]),
                   minimizer_kwargs={'args': (slice_ipol, fragment_ipol, slice_grad_ipol,
                     fragment_grad_ipol, slice)})
"""
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