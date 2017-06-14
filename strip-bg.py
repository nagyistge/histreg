#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-14
# Revision of the original segment-by-hue.py algorithm. Goal: fix improper
# segmentation on bluish-tinted brain slices.

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from args import *
import os
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage.filters import median_filter

# Command-line argument flags
CLFLAGS = {'tiff': '--tif',
           'dims': '--dims',
           'weight': '-w',
           'clusters': '-c',
           'demean': '--demean',
           'normvar': '--normvar',
           'show': '--show',
           'save': '--save',
           'mask': '--mask',
           'median': '--median',
           'invert': '--inv',
           'segval': '-s',
           'format': '-f'}

#  Layer identifiers
LAYERIDS = {'red': 'r',
            'green': 'g',
            'blue': 'b',
            'hue': 'h',
            'saturation': 's',
            'value': 'v',
            'Lightness': 'L',
            'LabA': 'A',
            'LabB': 'B'}

FMTS = {'BMP': 'bms', 'EPS': 'eps', 'GIF': 'gif', 'ICNS': 'icns', 'ICO': 'ico',
        'IM': 'im', 'JPEG': 'jpg', 'JPEG 2000': 'jpg', 'MSP': 'msp',
        'PCX': 'pcx', 'PNG': 'png', 'PPM': 'ppm', 'SGI': 'sgi',
        'SPIDER': 'spider', 'TIFF': 'tif', 'WebP': 'webp', 'XBM': 'xbm'}


def main():
    err = 0

    imfilelist = subarg(sys.argv[0])
    if imfilelist[0] == '':
        print ('No image was specified.')
        err = err + 1
        exit()

    for arg in sys.argv[1+len(imfilelist):]:
        if str(arg).startswith('-'):
            if not (arg in CLFLAGS.values()):
                print ('Invalid argument: {}'.format(arg))
                err = err + 1
                exit()

    # Get user-specified layers
    lset = []
    if argexist(CLFLAGS['dims']):
        if argexist(CLFLAGS['dims'], True):
            for l in list(subarg(CLFLAGS['dims'])[0]):
                if not (l in LAYERIDS.values()):
                    err = err + 1
                    print ('Invalid layer specification.')
                else:
                    lset.append(l)
        else:
            print ('Invalid layer specification.')
            err = err + 1
            exit()
    else:
        lset = ['auto']

    # Get weights for the feature layers
    weights = subarg(CLFLAGS['weight'], 'auto')
    if lset[0] != 'auto':
        if weights[0] != 'auto':
            if len(lset) != len(weights):
                err = err + 1
                print ("Weights don't match the input after {}.".
                       format(CLFLAGS['dims']))
                exit()
    try:
        if not (weights[0] == 'auto'):
            weights = np.array([float(weight) for weight in weights])
    except:
        print ('Invalid weight specification.')
        err = err + 1
        exit()

    # Read the number of clusters
    try:
        N_CLUSTERS = int(subarg(CLFLAGS['clusters'], 2)[0])
    except:
        print ('Invalid cluster number specification.')
        err = err + 1
        exit()

    # Read segmentation label
    try:
        tmp = subarg(CLFLAGS['segval'], 0)
        SEGVALS = np.array([int(segval) for segval in tmp])
        del(tmp)

    except:
        print ('Invalid segmentation label specification.')
        err = err + 1
        exit()

    # Read format
    fmt = subarg(CLFLAGS['format'], 'TIFF')[0].upper()
    if not (fmt in FMTS.keys()):
        print ('Invalid format specification. Supported formats: \n{}'
               .format(FMTS.keys()))
        err = err + 1
        exit()
    if argexist(CLFLAGS['tiff']) & (fmt != 'TIFF'):
        print ("When the {} switch is used, only the default 'TIF' format can be "
               "specified.".format(CLFLAGS['tiff']))
        err = err + 1
        exit()

    # Read median filtering kernel size
    if argexist(CLFLAGS['median']):
        if argexist(CLFLAGS['median'], True):
            try:
                medkernel = int(subarg(CLFLAGS['median'])[0])
            except:
                err = err + 1
                print ('Invalid median fileter kernel size.')
                exit()
        else:
            err = err + 1
            print ('A positive integer kernel size must be provided when the {} '
                   'switch is used.'.format(CLFLAGS['median']))
            exit()

    # Summarize task and insist on user confirmation before proceeding
    line = "\n{:d} file(s) will be processed:".format(len(imfilelist))
    print line
    print '=' * len(line) + '\n'
    print "\n".join(imfilelist)
    print "\nWould you like to start the process? [yes/no]: "
    if not confirmed_to_proceed():
        exit()

    # Process image(s)
    for imfile in imfilelist:
        # Validate file path
        if not os.path.isfile(imfile):
            err = err + 1
            print ('{} could not be recognised, therefore it was skipped.')
            continue

        fpath = os.path.split(imfile)[0]
        fname = os.path.split(imfile)[1]

        # Open file
        if argexist(CLFLAGS['tiff']):
            img = tiff.imread(imfile)
        else:
            img = np.array(Image.open(imfile))

        # Make sure that the image has 3 dimensions
        if not (len(img.shape) in (2,3)):
            err = err + 1
            print ('Invalid number of image dimensions in {}. The file was skipped.'
                   .format(imfile))
            continue
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
        h, w, ch = img.shape
        print ('Processing {}...'.format(imfile))
        print img.shape

        # Prepare layers to be used as dimensions for the k-Means classifier
        lstack = img
        if ch != 1:
            try:
                pilimg = Image.fromarray(img).convert('RGB')
            except:
                err = err + 1
                print ('Colour space conversion failed for {}. The image was '
                       'skipped.')
                continue
            hsv = lambda: np.asarray(pilimg.convert('HSV'))
            lab = lambda: np.asarray(pilimg.convert('LAB'))

            # Getter functions for the individual layers
            LAYERS = {LAYERIDS['red']: lambda: img[:, :, 0],
                      LAYERIDS['green']: lambda: img[:, :, 1],
                      LAYERIDS['blue']: lambda: img[:, :, 2],
                      LAYERIDS['hue']: lambda: hsv()[:, :, 0],
                      LAYERIDS['saturation']: lambda: hsv()[:, :, 1],
                      LAYERIDS['value']: lambda: hsv()[:, :, 2],
                      LAYERIDS['Lightness']: lambda: lab()[:, :, 0],
                      LAYERIDS['LabA']: lambda: lab()[:, :, 1],
                      LAYERIDS['LabB']: lambda: lab()[:, :, 2]}

            if not lset[0] == 'auto':
                lstack = []
                for l in lset:
                    lstack.append(LAYERS[l]())
                lstack = np.dstack(lstack)
        n_features = lstack.shape[-1]

        # Re-calculate weights
        if weights[0] == 'auto':
            weights = np.array([1] * n_features)
        weights = np.sqrt(weights / np.sum(weights).astype(np.float64))
        if len(weights) != n_features:
            err = err + 1
            print ("Weights did not match the number of layers in {}. The file was "
                   "skipped.".format(imfile))
            continue

        # Perform global k-Means classification of pixels
        if argexist(CLFLAGS['weight']):
            print 'Weights:', weights
            data = lstack.reshape(-1, n_features) * weights.reshape(1, n_features)
        else:
            data = lstack.reshape(-1, n_features)
        if argexist(CLFLAGS['demean']):
            data = data - np.mean(data, axis=0)
        if argexist(CLFLAGS['normvar']):
            data = data / np.var(data, axis=0)
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=1).fit(data)
        mask = np.asarray(kmeans.labels_, dtype=np.int8).reshape((h, w))
        if argexist(CLFLAGS['invert']):
            mask = np.max(mask) - mask
        if argexist(CLFLAGS['median'], True):
            mask = median_filter(mask, medkernel)
        segmentation = np.copy(img)
        segmentation[np.in1d(mask, SEGVALS).reshape(mask.shape)] = 0

        if argexist(CLFLAGS['save']):
            fn = os.path.join(fpath, fname[:-4])
            if argexist(CLFLAGS['tiff']):
                fnseg = fn + '_seg.tif'
                try:
                    tiff.imsave(fnseg, segmentation)
                    print ('SAVED: {}'.format(fnseg))
                except:
                    err = err + 1
                    print ('ERROR: {} could not be created.'.format(fnseg))
                fnmask = fn + '_mask.tif'
                if argexist(CLFLAGS['mask']):
                    try:
                        tiff.imsave(fnmask, mask.astype(np.uint8))
                        print ('SAVED: {}'.format(fnmask))
                    except:
                        err = err + 1
                        print ('ERROR: {} could not be created.'.format(fnmask))
            else:
                fnseg = fn+'_seg.'+FMTS[fmt]
                try:
                    Image.fromarray(segmentation).save(fnseg, fmt)
                    print ('SAVED: {}'.format(fnseg))
                except:
                    err = err + 1
                    print ('ERROR: {} could not be created.'.format(fnseg))
                if argexist(CLFLAGS['mask']):
                    fnmask = fn+'_mask.'+FMTS[fmt]
                    try:
                        Image.fromarray(mask).save(fnmask, fmt)
                        print ('SAVED: {}'.format(fnmask))
                    except:
                        err = err + 1
                        print ('ERROR: {} could not be created.'.format(fnmask))

        if argexist(CLFLAGS['show']):
            if argexist(CLFLAGS['mask']):
                plt.subplot(131)
                plt.imshow(img)
                plt.title('Original')
                plt.subplot(132)
                plt.imshow(mask, cmap='gray')
                plt.title('Mask')
                plt.subplot(133)
                plt.imshow(segmentation)
                plt.title('Segmentation')
            else:
                plt.subplot(121)
                plt.imshow(img)
                plt.title('Original')
                plt.subplot(122)
                plt.imshow(segmentation)
                plt.title('Segmentation')
            plt.show()

    if err == 0:
        print ('All tasks were successfully completed.')
    else:
        print ('The program encountered {:d} error(s).'.format(err))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        print ("""
        The strip-bg.py utility can be used to extract brain slices from images
        that also contain a background. The background must be visually 
        distinct from the brain tissue.
         
        Supported file formats: BMP, EPS, GIF, ICNS, ICO, IM, JPEG, JPEG 2000, 
        MSP, PCX, PNG, PPM, SGI, SPIDER, TIFF, WebP, XBM.
           
        The core process is an adjustable global k-Means classification, that 
        has the following options.
        
        Usage: ./strip-bg.py <input> [options]
        
        Options:
            --tif               Forces to use tifffile.py for opening and saving
                                images (PIL is used by default)
            --dims <rgbhsvLAB>  Specifies which colour channel(s) to use
                                (default: all image channels)
            -w [w1,w2...wn]     Weights for the colour channels
                                (default: equal weights for all channels)
            -c                  Number of clusters (default: 2)
            --demean            Demean channel values
                                (default: no)
            --normvar           Normalise channel values by their variance
                                (default: no)
            --mask              Output the mask.
            --median <size>     Perform median filtering on the mask with fixed
                                kernel size. (default: no filtering)
            --inv               Invert the mask. (default: no)
            -s <s1,s2...sn>     Mask values used for segmentation
                                (default: 0)
            --show              Show output on screen. (default: no)
            --save              Save output to file. (default: no)
            -f      <format>    Output file format. See above what is supported.
        """)