#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-12
# Imports a GM probability map and a respective sampling map and performs
# regional normalisation, finds connected components filters out small patches
# (thresholding by area).

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from args import *
import os
from PIL import Image


CLFLAGS = {'smap': '--smap',    # load sampling map from file
           'draw': '--draw',    # draw sampling map: fh, fw, sh, sw
           'thresh': '--thresh',      # fixed threshold value for binarisation (0-1)
           'area': '--area',    # area threshold for small patches
           'cpu': '--cpu',      # use n number of cores
           'show': '--show',    # show results
           'save': '--save',    # save cleaned-up image, optionally specify name
           'inv': '--inv',      # invert binary output
           'tiff': '--tif',     # force opening by tifffile.py
           'format': '-f'}      # specify output format

def draw_frames(h, w, fh, fw, sh, sw):
    parcellation = np.zeros((h, w))
    for row in range((h-fh)/sh + 1):
        for col in range((w-fw)/sw + 1):
            #print "row={:d}/{:.0f}, col={:d}/{:.0f}".format(row+1, (h-fh)/sh + 1, col+1, (w-fw)/sw + 1)
            top = int(row * sh)
            left = int(col * sw)
            parcellation[top:top+fh, left:left+fw] = parcellation[top:top+fh, left:left+fw] + 1
    return parcellation


imfilelist = subarg(sys.argv[0])
smaplist = subarg(CLFLAGS['smap'])

if len(smaplist) != len(imfilelist):
    if len(smaplist) == 0:
        if not argexist(CLFLAGS['draw'], True):
            print ('Sampling map was not specified.')
            exit()
    if len(smaplist) == 1:
        smaplist = [smaplist[0]] * len(imfilelist)
    else:
        print ('The number of probability and sampling maps did not match.')
        exit()

# Summarize task and insist on user confirmation before proceeding
line = "\n{:d} files will be processed:".format(len(imfilelist))
print line
print '=' * len(line) + '\n'
print "\n".join(imfilelist)
line = "\nSelected sampling maps ({}):".format(len(smaplist))
print line
print '=' * len(line) + '\n'
print "\n".join(smaplist)

print "\nWould you like to start the process? [yes/no]: "
if not confirmed_to_proceed():
    exit()

smaplist = iter(smaplist)

for imfile in imfilelist:
    smapfile = smaplist.next()
    fpath = os.path.split(imfile)[0]
    fname = os.path.split(imfile)[1]

    # Load the probability map
    try:
        if argexist(CLFLAGS['tiff']):
            probmap = tiff.imread(imfile)
        else:
            probmap = np.array(Image.open(imfile))
        h, w = probmap.shape
        print 'Opening {}...'.format(imfile)
    except:
        print ('Probability map could not be read from {} or had invalid '
               'dimensions.'.format(imfile))
        continue

    # Load or draw the sampling map
    if argexist(CLFLAGS['smap'], True):
        try:
            smap = tiff.imread(smapfile).astype(np.int64)
            h1, w1 = smap.shape
            if (h1 != h) | (w1 != w):
                print ('Probability and sampling map dimensions did not match.')
                continue
        except:
            print ('Sampling map could not be read from {} or had invalid '
                   'dimensions.'.format(smapfile))
            continue
    elif argexist(CLFLAGS['draw'], True):
        try:
            fh, fw, sh, sw = subarg(CLFLAGS['draw'])
            smap = draw_frames(h, w, int(fh), int(fw), int(sh), int(sw))
        except:
            print ('Drawing the sampling map was not successful.')
            continue
    else:
        print ('Either {} or {} must be specified.'.format(CLFLAGS['smap'],
                                                           CLFLAGS['draw']))
        exit()

    # Create unique regional IDs
    row = smap[0,:]
    col = smap[:,0]
    col_ord = np.cumsum(np.abs(col-np.roll(col, 1))) + 1
    row_ord = np.cumsum(np.abs(row-np.roll(row, 1))) + 1
    idmap = np.max(row_ord) * np.dot((col_ord-1).reshape(-1, 1), np.ones(row_ord.shape).reshape(1,-1)) + row_ord

    # Normalize all cells
    largest = np.max(idmap)
    newimg = np.copy(probmap)

    for idx in np.unique(idmap): #xrange(1, int(largest)+1, 1):
        tmp = np.array(newimg[idmap==idx], dtype=np.float32)
        # Since the values in the smap file might be scaled, not all IDs will
        # exist.
        if tmp.size == 0:
            continue
        maxp = np.max(tmp)
        if maxp != 0:
            tmp = np.nan_to_num((tmp - np.min(tmp)) / (np.max(tmp)-np.min(tmp)))  # new edit
        newimg[idmap == idx] = tmp

    # Binarise the normalised image (Otsu thresholding)
    if argexist(CLFLAGS['thresh'], True):
        try:
            th = float(subarg(CLFLAGS['thresh'])[0])
        except:
            print ('Invalid threshold value. Please set it to 0 <= th <= 1.')
            exit()
    else:
        th = threshold_otsu(newimg)
    newimg[newimg >= th] = 255
    newimg[newimg < th] = 0

    # Get rid of the small patches
    labels = label(newimg, connectivity=1)
    regions = regionprops(labels)
    areas = np.asarray([region['area'] for region in regions])

    cleanimg = np.copy(newimg)
    try:
        area_limit = int(subarg(CLFLAGS['area'], 1000)[0])
    except:
        print ('Invalid number for area limit.')
        exit()
    cleanimg[np.in1d(labels, np.asarray(
        [region['label'] for region in regions if
         region['area'] < area_limit])).reshape(cleanimg.shape)] = 0

    # Invert output if asked so
    if argexist(CLFLAGS['inv']):
        cleanimg = np.max(cleanimg) - cleanimg

    if argexist(CLFLAGS['show']):
        plt.subplot(131)
        plt.imshow(probmap, cmap='gray')
        plt.title('Original')
        plt.subplot(132)
        plt.imshow(newimg, cmap='gray')
        plt.title('Regional re-norm')
        plt.subplot(133)
        plt.imshow(cleanimg, cmap='gray')
        plt.title('Cleaned-up')
        plt.show()

    if argexist(CLFLAGS['save']):
        fmt = subarg(CLFLAGS['format'], 'tif')[0]
        fn = os.path.join(fpath, fname[:-4] + '_cleanup.{}'.format(fmt))
        savearg = subarg(CLFLAGS['save'])
        if (len(savearg) == len(imfilelist)) & (savearg[0] != ''):
            fn = savearg
        if argexist(CLFLAGS['tiff']) & (fmt.lower() == 'tif'):
            tiff.imsave(fn, cleanimg.astype(np.uint8))
        else:
            Image.fromarray(cleanimg.astype(np.uint8), 'L').save(fn, fmt)
        print ('SAVE: {}.'.format(fn))
