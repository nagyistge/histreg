#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-14
# Fork of the original insert-block.py. Goal: make it a user-friendly utility
# capable of registering multi-channel 2D images using rigid-body transformation

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos, pi, floor, sqrt
from fractions import Fraction
from args import *
import os
from PIL import Image

CLFLAGS = {'ref': '--ref',
           'rgb': '--rgb',
           'apply': '--applyxfm',
           'initvals': '--initvals',
           'show': '--show',
           'out': '--out',
           'omat': '--omat',
           'tif': '--tif',
           'ds': '--ds',
           'verbose': '-v'}

N_AFF_PARAMS = 4


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ## FUNCTION DEFINITIONS ## #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def extract_shape(img):
    """Additional flexibility to work with multi-channel images."""
    s = img.shape
    if len(s) == 2:
        s = s + (1,)
    if len(s) == 3:
        pass
    else:
        print ('Invalid dimensions for image')
        raise ValueError
    return s


def downsample(img, factor, bounds_error=False, fill_value=0):
    """Downsample images isotropically by a given scalar. Uses interpolation."""
    h, w, ch = extract_shape(img)
    dtype = img.dtype
    nx = np.linspace(0, w, int(floor(w / factor)) + 1, False)
    ny = np.linspace(0, h, int(floor(h / factor)) + 1, False)
    nyy, nxx = np.meshgrid(ny, nx, indexing='ij')
    newimg = np.zeros((ny.size, nx.size, ch), dtype=dtype)
    for c in range(ch):
        if ch == 1:
            ipol = RegularGridInterpolator((range(h), range(w)), img[:, :],
                                           bounds_error=bounds_error,
                                           fill_value=fill_value)
        else:
            ipol = RegularGridInterpolator((range(h), range(w)), img[:, :, c],
                                       bounds_error=bounds_error,
                                       fill_value=fill_value)
        newimg[:, :, c] = ipol(np.vstack((nyy.ravel(), nxx.ravel())).T) \
            .reshape(nxx.shape).astype(dtype)
    if ch == 1:
        return newimg[:, :, 0]
    else:
        return newimg


def colour_grad(img, kernelsize=3):
    """Calculates local multi-channel gradients using my harmonic kernel."""
    h, w, ch = extract_shape(img)
    harmonic = lambda n: sum(Fraction(1, d) for d in range(1, n + 1))
    U = np.ones((1, kernelsize))
    D = 2.0 / (kernelsize * (kernelsize - 1)) * \
        np.array([float(harmonic(i) - harmonic(kernelsize - 1 - i))
                  for i in range(kernelsize)])
    offset = int(kernelsize / 2)
    dx = np.zeros((h,w,ch), dtype=np.float64)
    dy = np.zeros((h,w,ch), dtype=np.float64)
    for c in range(ch):
        for y in xrange(offset, h - (kernelsize - offset)):
            #print '{}/{} row...'.format(y - offset + 1, h - kernelsize)
            for x in xrange(offset, w - (kernelsize - offset)):
                if ch == 1:
                    K = img[y - offset:y - offset + kernelsize,
                            x - offset:x - offset + kernelsize]
                else:
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


def get_affine_from_params(affine_params, orig_x=0, orig_y=0, factor=1):
    """Creates 3x3 affine matrix from the supplied 4 parameters.
    Please note the constraint of isometric scaling. The factor compensates for
    downscaling."""
    rot_z, dx, dy, sx = affine_params
    sy = sx
    dx = factor*(dx - orig_x)
    dy = factor*(dy - orig_y)
    affine = np.array([[sx * cos(rot_z), -sy * sin(rot_z),
                        dx * sx * cos(rot_z) - dy * sy * sin(rot_z) +
                        factor*orig_x],
                       [sx * sin(rot_z), sy * cos(rot_z),
                        dx * sx * sin(rot_z) + dy * sy * cos(rot_z) +
                        factor*orig_y],
                       [0, 0, 1]])
    return affine

def transform(img, affine, refimg=None):
    """Performs affine transformation on the coordinates of an image given the
    transformation matrix. Output coordinates are based on the coordinates of a
    reference image and are in the form array([Y, X]).

    If no reference image is specified, the transformation is self-referential.
    The rotation is relative to the point (orig_x, orig_y). If either is not
    specified, they are set to the respective component of the centre of
    gravity."""

    if refimg is not None:
        h, w, ch = extract_shape(refimg)
    else:
        h, w, ch = extract_shape(img)
    yy, xx = np.meshgrid(range(h), range(w), indexing='ij')
    res = np.dot(np.linalg.inv(affine), np.vstack((xx.ravel(), yy.ravel(),
                                                   np.ones(xx.size))))
    return np.roll(res[:2, :].T, 1, axis=1)


def transform_img(img, affine, refimg=None):
    """Performs actual image transformation based on the location of the origin
    and the components of an affine matrix."""
    h, w, ch = extract_shape(img)
    dtype = img.dtype
    ipol = RegularGridInterpolator((range(h), range(w)), img,
                                   bounds_error=False,
                                   fill_value=np.array([0]*ch))
    if refimg is not None:
        h, w, ch = extract_shape(refimg)
    transformed_img = ipol(transform(img, affine, refimg=refimg))\
                      .reshape((h,w,ch)).astype(dtype)
    if ch == 1:
        return transformed_img[:, :, 0]
    else:
        return transformed_img


def set_interpolators(img, refimg, gradients=False):
    """Creates interpolators for both the input and the reference images. Also
    creates interpolators for the multi-channel gradient maps (optional).

    The interpolators bridge the gap between the potentially different sizes of
    the input and the reference image. Setting up the interpolators outside the
    cost function improves the performance of registration."""

    h, w, ch = extract_shape(refimg)
    ih, iw, ich = extract_shape(img)

    # Define interpolators for both the input and the reference image
    refimg_ipol = RegularGridInterpolator((range(h), range(w)), refimg,
                                          bounds_error=False,
                                          fill_value=np.array([0] * ch))
    img_ipol = RegularGridInterpolator((range(ih), range(iw)), img,
                                       bounds_error=False,
                                       fill_value=np.array([0] * ich))

    # Define interpolators for the colour gradient maps
    if gradients:
        img_grad_ipol = RegularGridInterpolator((range(ih), range(iw)),
                                                colour_grad(img, 10),
                                                bounds_error=False,
                                                fill_value=np.array(
                                                    [0] * ich))
        refimg_grad_ipol = RegularGridInterpolator((range(h), range(w)),
                                                   colour_grad(refimg, 10),
                                                   bounds_error=False,
                                                   fill_value=np.array(
                                                       [0] * ch))
        return img_ipol, refimg_ipol, img_grad_ipol, refimg_grad_ipol
    else:
        return img_ipol, refimg_ipol


def costfun(affine_params, img_ipol, refimg_ipol, img_grad_ipol=None,
            refimg_grad_ipol=None, orig_x=0, orig_y=0, verbose=False):
    """Calculates the alignment penalty score (cost) between the input image and
    the reference image given the transformation parameters.

    The total cost is the sum of squared differences plus a scalar multiple of a
    regularisation term, which is currently a sum of squared differences between
    multi-channel gradients."""

    alpha = 1  # the scalar

    affine = get_affine_from_params(affine_params, orig_x, orig_y, 1)
    yx_xfm = transform(img=img_ipol.values, affine=affine,
                       refimg=refimg_ipol.values)
    yy, xx = np.meshgrid(refimg_ipol.grid[0], refimg_ipol.grid[1],
                         indexing='ij')
    yx_ref = np.vstack((yy.ravel(), xx.ravel())).T

    h, w, ch = extract_shape(refimg_ipol.values)
    lsqterm = np.sum((refimg_ipol(yx_ref).reshape((h, w, ch)) -
                      img_ipol(yx_xfm).reshape((h, w, ch))) ** 2)
    regterm = np.sum((refimg_grad_ipol(yx_ref).reshape((h, w, ch)) -
                      img_grad_ipol(yx_xfm).reshape((h, w, ch))) ** 2)
    cost = lsqterm + alpha * regterm
    if verbose:
        print cost

    return cost


def perform_registration(img, refimg, affine_param_bounds, init_steps=5,
                         dscale=1, initvals=None, verbose=True):
    """Perform registration given the input and the reference image plus the
    bounds on the affine parameters and the number of initialisation steps for
    each affine parameter."""

    # Set up interpolators
    ipols = set_interpolators(downsample(img, dscale),
                              downsample(refimg, dscale), gradients=True)

    # Calculate the centre of gravity for the input image
    h, w, ch = extract_shape(ipols[0].values)
    if ch > 1:
        orig_y, orig_x = \
            np.mean(np.vstack(np.where(np.sum(ipols[0].values, axis=2)!=0)).T,
                    axis=0)
    else:
        orig_y, orig_x = np.mean(np.vstack(np.where(ipols[0].values != 0)).T,
                                 axis=0)

    # Initialise registration
    if initvals is None:
        gridpts = tuple([np.linspace(affine_param_bounds[i][0],
                                     affine_param_bounds[i][1],
                                     init_steps, endpoint=True)
                        for i in range(len(affine_param_bounds))])
        grid = np.meshgrid(*gridpts)
        grid = np.vstack([grid[i].ravel() for i in range(len(grid))]).T
        initial_costs = []
        print 'Calculating initial guess...'
        for x0 in grid:
            initial_costs.append(costfun(x0, img_ipol=ipols[0],
                                         refimg_ipol=ipols[1],
                                         img_grad_ipol=ipols[2],
                                         refimg_grad_ipol=ipols[3],
                                         orig_x=orig_x, orig_y=orig_y,
                                         verbose=False))
        x0 = grid[np.argmin(np.vstack(initial_costs))]
        print 'Best guess for initialisation: ', x0
    else:
        x0 = initvals
        x0[1] = x0[1] * 1.0 / dscale
        x0[2] = x0[2] * 1.0 / dscale
    print 'Scaled initial affine parameters:', x0

    # Perform registration by minimising the cost function
    print 'Optimizing...'
    opt = minimize(costfun, x0=x0, bounds=affine_param_bounds,
                   args=(ipols[0], ipols[1], ipols[2], ipols[3], orig_x, orig_y,
                         verbose))

    # Generate output: transformed image and transformation matrix
    print opt.x
    omat = get_affine_from_params(opt.x, orig_x=orig_x, orig_y=orig_y,
                                  factor=dscale)
    out = transform_img(img, omat, refimg)

    return out, omat


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#      ## MAIN CODE ##       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def main():
    err = 0

    # Load input image
    imfile = subarg(sys.argv[0])[0]
    fpath, fname = os.path.split(imfile)
    try:
        if argexist(CLFLAGS['tif']):
            img = tiff.imread(imfile)
        else:
            img = np.array(Image.open(imfile))
        if argexist(CLFLAGS['rgb']) & (extract_shape(img)[2] > 3):
            img = img[:,:,:3]
    except:
        print ('Input image could not be opened from {}.'.format(imfile))
        exit()

    # Read reference
    if argexist(CLFLAGS['ref']):
        if argexist(CLFLAGS['ref'], True):
            refimfile = subarg(CLFLAGS['ref'])[0]
            try:
                if argexist(CLFLAGS['tif']):
                    refimg = tiff.imread(refimfile)
                else:
                    refimg = np.array(Image.open(refimfile))
                if argexist(CLFLAGS['rgb']) & (extract_shape(refimg)[2] > 3):
                    refimg = refimg[:, :, :3]
            except:
                print ('Reference image could not be opened from {}.'
                       .format(refimfile))
                exit()
        else:
            print ('Reference image was not specified.')
            exit()

    # Check image compatibility
    if extract_shape(img)[2] != extract_shape(refimg)[2]:
        print ('Channel number mismatch between input and reference.')
        exit()

    # Read downsample factor
    try:
        factor = float(subarg(CLFLAGS['ds'], 1)[0])
    except:
        print ('Invalid factor for downsampling.')
        exit()

    # Read transformation matrix from file if specified
    mat = None # this will remain None if registration is needed
    if argexist(CLFLAGS['apply']):
        matfile = subarg(CLFLAGS['apply'])[0]
        try:
            mat = np.loadtxt(matfile, dtype=np.float64)
        except:
            print ('Transformation matrix could not be loaded from {}'
                   .format(matfile))
            exit()
        if mat.shape != (3,3):
            print ('Transformation matrix had invalid shape.')
            exit()

    # Read initialisation values
    if argexist(CLFLAGS['initvals']):
        if argexist(CLFLAGS['initvals'], True):
            initvals = subarg(CLFLAGS['initvals'])
            if len(initvals) != N_AFF_PARAMS:
                print ('Invalid affine parameters for initialisation.')
                exit()
            try:
                # Strip brackets
                initvals[0] = initvals[0][1:]
                initvals[-1] = initvals[-1][:-1]
                initvals = np.array([float(val) for val in initvals])
            except:
                print ('Invalid affine parameters for initialisation.')
                exit()
        else:
            print ('Initial affine parameters are not specified.')
            exit()

    # Read output name
    outfile = None
    if argexist(CLFLAGS['out']):
        outfile = subarg(CLFLAGS['out'], os.path.join(fpath, fname[:-4] +
                                                      '_aligned' +
                                                      fname[-4:]))[0]
    omatfile = None
    if argexist(CLFLAGS['omat']):
        omatfile = subarg(CLFLAGS['omat'], os.path.join(fpath, fname[:-4] +
                                                      '_omat.mat'))[0]

    # Read verbose switch
    if argexist(CLFLAGS['verbose']):
        verbose = True
    else:
        verbose = False

    # Do the job
    if mat is None:
        print ('REGISTRATION MODE active.')
        bnds = ((-pi, pi), (-100, 100), (-100, 100), (0.8, 1.2))
        outimg, omat = perform_registration(img, refimg,
                                            affine_param_bounds=bnds,
                                            init_steps=5, dscale=factor,
                                            initvals=initvals, verbose=verbose)

        if argexist(CLFLAGS['show']):
            print ('Showing the alignment...')
            plt.imshow((outimg.astype(np.float64)-refimg.astype(np.float64)))
            plt.show()

    else:
        print ('TRANSFORMATION MODE active.')
        try:
            outimg = transform_img(img, mat, refimg)

            if argexist(CLFLAGS['show']):
                print ('Showing the transformed image...')
                plt.imshow(outimg.astype(img.dtype))
                plt.show()
        except:
            print ('ERROR: The transformation was not successful.')
            err = err + 1

    # Save the output
    if outfile is not None:
        try:
            if argexist(CLFLAGS['tif']):
                tiff.imsave(outfile, outimg.astype(img.dtype))
            else:
                Image.fromarray(outimg).save(outfile)
            print ('SAVED: {}'.format(outfile))
        except:
            print ('ERROR: {} could not be saved.'.format(outfile))
            err = err + 1
    if omatfile is not None:
        try:
            np.savetxt(omatfile, omat)
            print ('SAVED: {}'.format(omatfile))
        except:
            print ('ERROR: {} could not be saved.'.format(omatfile))
            err = err + 1

    # Conclude run
    if err > 0:
        print ('{} error(s) occured.'.format(err))
    else:
        print ('All tasks were successfully completed.')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Program execution starts here. #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print (
    """
    The register-img.py utility performs rigid-body registration between two 2D
    images. Given a prespecified affine matrix, a linear transformation is 
    carried out on the input image.
    
    When the program is run to register images, a transformed image and/or the
    transformation matrix is returned. The size and shape of the output image 
    are equal to the size and shape of the reference image.
    When the program is run to perform a prespecified linear transformation, the
    resultant image is returned.
    
    Usage:
        ./register-img.py <input> --ref <reference> --out [output] --omat
        ./register-img.py <input> --applyxfm <affine.mat> --out [output]
        
    Options:
        --ds <factor>       Downsample images for better performance. Factor: px/mm.
                            (Default: off. Recommended downsampling: 1mm/px)
        --tif               Forces to use tifffile.py for the input and output.
        --show              Show the output.
        --initvals          Manual initialisation of affine parameters. Use []!
          [rz,dx,dy,sxy]    (rz: rotation, dx, dy: translation, sxy: scale)
                            (Default: automatic best-guess initialisation.)
        --rgb               Forces to use maximum 3 channels. (Use for RGBA.)
        -v                  Verbose: report the evolution of the cost function.
    """
        )