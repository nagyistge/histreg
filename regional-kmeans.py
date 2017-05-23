#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# Commits
# 23 May 2017: not fully tested but works for NP-14-14 (97)_seg.tif with alpha channel and edt

import numpy as np
from sklearn.cluster import KMeans
from args import *
import re
import fnmatch
from PIL import Image
from math import sqrt, floor
from exceptions import *
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import distance_transform_edt
import os

# Command-line flags are referenced by their name
CLFLAGS = {'subdir': '-r',
           'exclude': '--exclude',
           'include': '--include',
           'mask': '-m',
           'clusters': '-c',
           'frame': '-f',
           'step': '-s',
           'dims': '--dim',
           'edt': '--edt',
           'cpu': '--cpu',
           'smap': '-b',
           'mlth': '--no-mlth',
           'probs': '--probs',
           'verbose': '-v'}

# PIL colour modes for grayscale and colour images
GMODES = {'1', 'L', 'P', 'I', 'F', 'LA'}
CMODES = {'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'RGBX', 'RGBa'}

#  Layer identifiers
LAYERIDS = {'red': 'r',
            'green': 'g',
            'blue': 'b',
            'hue': 'h',
            'saturation': 's',
            'value': 'v',
            'Lightness': 'L',
            'LabA': 'A',
            'LabB': 'B',
            'edt': 'd'}

SEGTAG = '_mlth'
PROBTAG = '_class'
SMAPTAG = '_smap'
MASKTAG = '_mask'


# Check for invalid (non-numerical) values in an image (numpy array)
# and return their indices if there is at least one.
def good_img_values(img):
    nan = np.flatnonzero(np.isnan(img))
    inf = np.flatnonzero(np.isinf(img))
    neginf = np.flatnonzero(np.isneginf(img))
    if nan.size + inf.size + neginf.size > 0:
        return False, nan, inf, neginf
    else:
        return True


def set_framesize(h, w, stepsize=None):
    # if the step sizes haven't yet been set
    if stepsize is None:
        st = floor(sqrt(h * w * 0.003))  # the constant: one isometric step should cover 0.3% of the total image area
        n = 10  # try to optimize for 10 steps
        # decrease the number of steps until both the height and width of the frame are valid
        while ((h - n * st < 0) | (w - n * st < 0)) & (n > 1):
            n = n - 1
        if (n == 1) & ((h - n * st < 0) | (w - n * st < 0)):
            raise FrameError('Automatic determination of frame size failed. Try to set it manually.')
        return int(h - n * st), int(w - n * st)

    # if both step sizes have already been set
    # I assume the integrity of stepsize has already been checked
    else:
        sh, sw = tuple(stepsize)
        if (sh >= h) | (sw >= w):
            raise FrameError('At least one step size is larger than the image.')
        n = 10
        while ((h * 1.0 / sh < n) | (w * 1.0 / sw < n)) & (n > 1):
            n = n - 1
        if (n == 1) & ((h * 1.0 / sh < n) | (w * 1.0 / sw < n)):
            raise FrameError('Automatic determination of frame size failed. Adjust step sizes.')
        return int(h - n * sh), int(w - n * sw)


def set_stepsize(h, w, framesize=None):
    # if the frame sizes haven't yet been set
    if framesize is None:
        st = floor(sqrt(h * w * 0.003))
        fh, fw = 0, 0
        while ((fh <= 0) | (fw <= 0)) & (st > 1):
            try:
                fh, fw = set_framesize(h, w, (st, st))
            except FrameError:
                st = floor(st * 0.9)
        if (st <= 1) & ((fh <= 0) | (fw <= 0)):
            raise FrameError('Automatic determination of step sizes failed. Try to set them manually.')
        return int(st), int(st)

    # if the frame size have already been set
    else:
        fh, fw = tuple(framesize)
        if (fh >= h) | (fw >= w):
            raise FrameError('The frame exceeds the image in at least one dimension.')
        sw = floor(sqrt((w - fw) * 1.0 / (h - fh) * sqrt(h * w * 0.003)))
        sh = floor(sw * (h - fh) * 1.0 / (w - fw))
        sw = w / (w // sw)
        sh = h / (h // sh)
        if (sh < 1) | (sw < 1):
            raise FrameError('Automatic determination of step sizes failed. Adjust frame size.')
        return int(sh), int(sw)


def autocorrect_sizedef(sizearray):
    if len(sizearray) > 2:  # if too many entries
        corr = list(sizearray[:2])
    elif len(sizearray) < 2:  # if too few entries
        corr = ['auto']
    else:
        corr = list(sizearray)

    # Note that a new if statement guarantees that the truncated stepsize array gets checked, too.
    if len(corr) == 2:  # if exactly two entries - are they really integer numbers?
        try:
            corr = [int(s) for s in corr]
        except ValueError:
            corr = ['auto']
    return corr


def create_edt(mask, darkpixels, filter=False, area_thresh=1):
    # It is assumed that the mask is already binary.
    # Binarize dark pixel mask
    dp = np.copy(darkpixels)
    dp[dp > 0] = 1

    # Combine dark pixels with the background
    dp = np.logical_or(dp, 1 - mask / np.max(mask))

    # Filter out small patches
    if filter:
        edt_labels = label(dp)
        edt_regions = regionprops(edt_labels)
        l = np.asarray([region['label'] for region in edt_regions], dtype=np.int64)
        areas = np.asarray([region['area'] for region in edt_regions], dtype=np.int64)
        edt_labels[np.in1d(edt_labels, l[areas < area_thresh]).reshape(edt_labels.shape)] = 0
        newmask = np.copy(edt_labels)
        newmask[edt_labels > 0] = 1
        dp = newmask

    # Perform the Euclidean Distance Transform (EDT)
    edt = np.asarray(distance_transform_edt(1 - dp), dtype=np.float32)
    edt = 1 - edt / np.max(edt)

    return edt


def img_proc(imfile, maskfile=None, usealpha=False, layers='bgr', edtfile=None, n_clusters=3, framesize='auto',
             stepsize='auto'):
    # Initialise error sum
    err = 0

    if not os.path.isfile(imfile):
        print 'WARNING: Image at {} could not be recognised and therefore it was skipped.'.format(imfile)
        return 1

    # Load image and internal mask, if applies
    try:
        pilimg = Image.open(imfile)
    except IOError:
        print 'WARNING: Image at {} could not be opened and therefore it was skipped.'.format(imfile)
        return 1

    mask = None  # initialize mask variable
    # If it is grayscale, transform into RGB colour space
    if pilimg.mode in CMODES:
        # For RGBA images, use alpha channel as mask if no mask file has been specified
        if (pilimg.mode == 'RGBA') & usealpha:
            mask = np.array(pilimg)[:, :, -1]
            mask[mask != 0] = 1  # binarize mask
        img = np.array(pilimg.convert('RGB'))
    # If it is grayscale, transform into 8-bit format
    else:
        # Set image array data type so that it bypasses the bug in the k-Means implementation
        img = np.array(pilimg.convert('L'), dtype=np.float64)

    # Get dimensions and validate them
    if len(img.shape) == 2:
        h, w = img.shape
        ch = 1
    elif len(img.shape) == 3:
        h, w, ch = img.shape
    else:
        print 'WARNING: Image at {} had invalid dimensions and therefore it was skipped.'.format(imfile)
        return 1

    # Check the integrity of pixel values
    if not good_img_values(img):
        print 'WARNING: Image at {} had invalid pixel values and therefore it was skipped.'.format(imfile)
        return 1

    # Load external mask (if there is one and the alpha channel hasn't been used)
    if (maskfile is not None) & (mask is None):
        try:
            mask = Image.open(maskfile)
        except IOError:
            print ('WARNING: The mask for {} could not be loaded from {}. '
                   'The image was skipped.'.format(imfile, maskfile))
            return 1
        mask = np.array(mask.convert('L'))
        mask[mask != 0] = 1  # make sure that the mask is binary

    # If the user set no -m flag, create a pseudo-mask:
    if mask is None:
        mask = np.ones((h, w))

    # Prepare layers to be used as dimensions for the k-Means classifier
    # Getter functions for the data in different colour spaces
    # Adapt grayscale images to the syntax by channel multiplication
    rgb = lambda: np.repeat(img[:, :, np.newaxis], 3, axis=2) if ch == 1 else img
    hsv = lambda: rgb() if ch == 1 else np.asarray(pilimg.convert('HSV'))
    lab = lambda: rgb() if ch == 1 else np.asarray(pilimg.convert('LAB'))
    edt = lambda: np.array(Image.open(edtfile).convert('L'))

    # Getter functions for the individual layers
    LAYERS = {LAYERIDS['red']: lambda: rgb()[:, :, 0],
              LAYERIDS['green']: lambda: rgb()[:, :, 1],
              LAYERIDS['blue']: lambda: rgb()[:, :, 2],
              LAYERIDS['hue']: lambda: hsv()[:, :, 0],
              LAYERIDS['saturation']: lambda: hsv()[:, :, 1],
              LAYERIDS['value']: lambda: hsv()[:, :, 2],
              LAYERIDS['Lightness']: lambda: lab()[:, :, 0],
              LAYERIDS['LabA']: lambda: lab()[:, :, 1],
              LAYERIDS['LabB']: lambda: lab()[:, :, 2],
              LAYERIDS['edt']: lambda: edt()}

    # Create the batch of layers specified by the user
    lset = []
    neededt = False  # Flag: is it necessary to run two iterations to have an edtmap?
    for l in list(layers):
        try:
            lset.append(LAYERS[l]())
        except KeyError:
            print 'ERROR: Invalid layer identifier: {}. The program terminates.'.format(l)
            exit()
        except (IOError, AttributeError):
            if edtfile is not None:
                print ('EDT distance map could not be opened from {}, therefore the image at '
                       '{} was skipped.'.format(edtfile, imfile))
                exit()  # if the edtfile fails to load on the second iteration, this escapes the infinite loop
            else:
                print 'EDT distance map will be used in the second iteration.'
                neededt = True
                continue
    try:
        lset = np.dstack(lset).astype(np.float64)
    except ValueError:
        print ('ERROR: Layer shapes are not compatible. Please review the --dim setting. '
               'The program now terminates.')
        exit()

    n_features = lset.shape[2]

    # Set sliding frame parameters
    stepsize = autocorrect_sizedef(stepsize)
    framesize = autocorrect_sizedef(framesize)
    variablesampling = False  # Flag: need to output one sampling map per file (see manual)
    if stepsize[0] == 'auto':
        variablesampling = True
        stepsize = None
        if framesize[0] == 'auto':
            framesize = None
            stepsize = set_stepsize(h, w, framesize)
            framesize = set_framesize(h, w, stepsize)
        else:
            stepsize = set_stepsize(h, w, framesize)
    else:
        if framesize[0] == 'auto':
            variablesampling = True
            framesize = None
            framesize = set_framesize(h, w, stepsize)

    fh, fw = framesize
    sh, sw = stepsize

    # Create stack for classification results (layers: number of clusters + background)
    results = np.zeros((h, w, n_clusters + 1))
    if argexist(CLFLAGS['smap']):
        smap = np.zeros((h, w))

    # Slide window, perform k-Means classification and pool the results
    for row in range(int((h - fh) / sh) + 1):
        for col in range(int((w - fw) / sw) + 1):
            print "row={:d}/{:.0f}, col={:d}/{:.0f}".format(row + 1, (h - fh) / sh + 1, col + 1, (w - fw) / sw + 1)
            # Define window
            top = int(row * sh)
            left = int(col * sw)

            window = np.copy(lset[top:top + fh, left:left + fw, :])
            mask_window = np.copy(mask[top:top + fh, left:left + fw])

            # Extract (discard zero pixels), demean and normalize data

            data_all = window.reshape((-1, n_features), order='F')
            data = data_all[(mask_window > 0).reshape(-1, order='F'), :]
            data = data - np.mean(data, axis=0)
            data = data / np.var(data, axis=0)  # division by zero results in 0 (numpy default)

            # Perform k-Means clustering (prepare for hitting the bug)
            try:
                data = data.astype(np.float64)  # To avoid a bug in the k-means implementation.
                kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(data)

                # Reintroduce zero-pixels and reshape labels array
                # The axis swap is a trick to comply with earlier Fortran-order reshapes.
                labels = np.rollaxis(np.zeros_like(mask_window), 1, 0)
                mask_window = np.rollaxis(mask_window, 1, 0)
                labels[mask_window > 0] = 1 + kmeans.labels_  # 0 is for background
                labels = np.rollaxis(labels, 1, 0)

                # Standardise labels
                # 1: dark pixels, 2: GM, 3: WM
                order = 1 + np.argsort(kmeans.cluster_centers_[:, 0])  # order labels by the first feature values
                if not all(order[i] <= order[i + 1] for i in xrange(len(order) - 1)):
                    tmp = np.copy(labels)
                    for label in range(1, n_clusters, 1):
                        labels[tmp == label] = 1 + np.where(order == label)[0][0]

            # If the classifier hits the bug, set all labels to -1, so they won't be counted.
            except IndexError:
                print 'IndexError'
                err = err + 1
                labels = np.rollaxis(np.zeros_like(mask_window), 1, 0)
                mask_window = np.rollaxis(mask_window, 1, 0)
                labels[mask_window > 0] = -1
                labels = np.rollaxis(labels, 1, 0)
                continue

            # Stack classification results
            for clust in range(n_clusters + 1):
                inc = np.zeros_like(labels)
                inc[np.where(labels == clust)] = 1
                results[top:top + fh, left:left + fw, clust] = results[top:top + fh, left:left + fw, clust] + inc

            # Update sampling map (if necessary)
            if argexist(CLFLAGS['smap']):
                smap[top:top + fh, left:left + fw] = smap[top:top + fh, left:left + fw] + 1

    # Save results table as numpy array
    np.save('clustertable', results)

    # Generate output
    sums = np.sum(results, axis=2)
    probs = np.asarray(results, dtype=np.float32) / sums[:, :, np.newaxis]
    segmentation = np.argmax(probs, axis=2)
    checksum = np.sum(probs, axis=2)
    print "Checksum (=1 =1): {}, {}".format(np.min(checksum), np.max(checksum))

    p = os.path.split(imfile)
    fn = '.'.join(p[-1].split('.')[:-1])
    if len(fn) == 0:  # fn would be invalid for extensionless hidden files
        fn = p[-1]
    fp = p[0]
    if argexist(CLFLAGS['verbose']):
        fn = fn + "_c{}_f[{}-{}]_s[{}-{}]_dim[{}]]".format(n_clusters, framesize[0], framesize[1],
                                                           stepsize[0], stepsize[1], layers)

    # If a second iteration is needed because of a missing EDT distance map
    if neededt:
        # Take the probability map of the assumed dark pixels
        # Assumption: the dark pixels have the lowest intensity next to the background
        #  along the first specified dimension in --dim.
        # This assumption works well for bgr, but not for hsv. Please watch out for this!
        darkpixels = np.array(probs[:, :, 1])
        edt = create_edt(mask, darkpixels, filter=True, area_thresh=int(h * w * 0.00003))

        # Save EDT distance map and add it as input for the next iteration
        # The image must be saved as float, which is not supported by png, so tiff is used here
        Image.fromarray(edt, 'F').save(os.path.join(fp, fn + '_edt.tif'))
        print 'SAVE: {} has been saved.'.format(os.path.join(fp, fn + '_edt.tif'))
        edtfile = os.path.join(fp, fn + '_edt.tif')

        # Perform second iteration before saving the intermediate files
        img_proc(imfile, maskfile, usealpha, layers, edtfile, n_clusters, framesize, stepsize)
        # don't finish first iteration:
        if err != 0:
            return 1
        else:
            return 0

    # Write output to file(s)
    if not argexist(CLFLAGS['mlth']):
        try:
            Image.fromarray(np.asarray(segmentation, dtype=np.uint32), 'I').save(os.path.join(fp, fn + SEGTAG + '.tif'))
            print 'SAVE: {} has been saved.'.format(os.path.join(fp, fn + SEGTAG + '.tif'))
        except:
            err = err + 1
            print 'ERROR while writing file: {}'.format(os.path.join(fp, fn + SEGTAG + '.tif'))
    if argexist(CLFLAGS['probs']):
        for clust in range(n_clusters + 1):
            try:
                Image.fromarray(np.asarray(probs[:, :, clust], dtype=np.float32), 'F').save(os.path.join(fp, fn + PROBTAG + '{0:02d}.tif'.format(clust)))
                print 'SAVE: {} has been saved.'.format(os.path.join(fp, fn + PROBTAG + '{0:02d}.tif'.format(clust)))
            except:
                err = err + 1
                print 'ERROR while writing file: {}'.format(os.path.join(fp, fn + PROBTAG + '{0:02d}.tif'.format(clust)))
    if argexist(CLFLAGS['smap']):
        try:
            Image.fromarray(np.asarray(smap, dtype=np.uint32), 'I').save(os.path.join(fp, fn + SMAPTAG + '.tif'))
            print 'SAVE: {} has been saved.'.format(os.path.join(fp, fn + SMAPTAG + '.tif'))
        except:
            err = err + 1
            print 'ERROR while writing file: {}'.format(os.path.join(fp, fn + SMAPTAG + '.tif'))

    # Was the full processing of the current image without errors?
    if err != 0:
        return 1
    else:
        return 0


def main():
    # Define input path(s), where files will be searched
    inpath = subarg(sys.argv[0], default_value=os.getcwd())  # default to the current working directory
    if len(inpath) > 1:
        print 'Multiple paths were found: processing {} of them.'.format(len(inpath))
    elif inpath == os.getcwd():
        print 'WARNING: input path not specified. The current working directory will be used:\n', inpath[0]

    # Define inclusion/exclusion filters
    INC = subarg(CLFLAGS['include'], '*.tif,*.tiff')  # default to TIFF files
    EXC = subarg(CLFLAGS['exclude'])

    if argexist(CLFLAGS['exclude'], subarg=True):
        # if only --exclude is specified, automatically include all files
        if not argexist(CLFLAGS['include'], subarg=True):
            INC = '*'  # note that *.* would exclude files without extension

    # Exclude user-defined file names
    if argexist(CLFLAGS['mask'], subarg=True):
        for maskfile in subarg(CLFLAGS['mask']):
            EXC.append(maskfile)

    # Collect all files that should be processed
    imglist = []
    for fp in inpath:
        # If the current path is a directory
        if os.path.isdir(fp):
            if fp == '.':
                fp = os.getcwd()
            for path, subdirs, files in os.walk(fp, topdown=True):
                # Exclude subdirectories unless specified
                if not argexist('-r'):
                    subdirs[:] = []
                for file in files:
                    fn = '.'.join(file.split('.')[:-1])
                    if len(fn) == 0:  # fn would be invalid for extensionless hidden files
                        fn = file
                    # Exclude hidden files, files with output tags, apply user-set inclusion/exclusion filters
                    if (fn[0] != '.') & \
                            (not fn.lower().endswith(SEGTAG)) & \
                            (not fn.lower().endswith(SMAPTAG)) & \
                            (not fn.lower().endswith(MASKTAG)) & \
                            (re.match(re.compile(PROBTAG + "[0-9][0-9]"), fn[-(2 + len(PROBTAG)):]) is None) & \
                            (any(fnmatch.fnmatch(file, p) for p in INC)) & \
                            (not any(fnmatch.fnmatch(file, p) for p in EXC)):
                        imglist.append(os.path.join(path, file))

        # If the current path points at a file
        elif os.path.isfile(fp):
            fn = '.'.join(fp.split('.')[:-1])
            if len(fn) == 0:  # fn would be invalid for extensionless hidden files
                fn = fp
            # Exclude hidden files, files with output tags, apply user-set inclusion/exclusion filters
            if (fn[0] != '.') & \
                    (not fn.lower().endswith(SEGTAG)) & \
                    (not fn.lower().endswith(SMAPTAG)) & \
                    (not fn.lower().endswith(MASKTAG)) & \
                    (re.match(re.compile(PROBTAG + "[0-9][0-9]"), fn[-(2 + len(PROBTAG)):]) is None) & \
                    (any(fnmatch.fnmatch(fp, p) for p in INC)) & \
                    (not any(fnmatch.fnmatch(fp, p) for p in EXC)):
                # Add local files with their full path, so that duplicates can be removed
                if os.path.split(fp)[0] == '':
                    fp = os.path.join(os.getcwd(), fp)
                imglist.append(fp)
        else:
            print 'WARNING: Path {} could not be recognised.'.format(fp)

    # Remove possible duplicates
    imglist = list(set(imglist))

    # Load binary mask(s)
    # If the user has set the -m argument
    if argexist(CLFLAGS['mask']):
        usealpha = False
        masks = subarg(CLFLAGS['mask'], MASKTAG)
        # If the user hasn't provided any subarguments to the -m flag,
        # create search list for the corresponding mask files.
        if masks[0] == MASKTAG:
            usealpha = True
            masks = [''.join(i.split('.')[:-1]) + MASKTAG + '.' + i.split('.')[-1] for i in imglist]
            print ('Binarized alpha channels will be used as masks where possible. '
                   'Files with a _mask tag will be used where the alpha channel is unavailable.')
        # If the user did specify a number of maskfiles, and
        # there is a total agreement between image(s) and mask(s):
        # (the elif and the order of conditions is crucial here)
        elif len(imglist) == len(masks):
            usealpha = False
            masks = [''.join(i.split('.')[:-1]) + MASKTAG + '.' + i.split('.')[-1] for i in imglist]
            print 'Individual masks with identical file names and a mask tag will be used where possible.'
        # If there is a discrepancy between the number of mask files and image files:
        else:
            if len(masks) == 1:
                print 'WARNING: The same mask will be used for all images.'
                masks = [masks[0] for _ in imglist]
            else:
                print ('ERROR: The number of masks and the number of images are not equal. '
                       'Please amend it. The program now terminates.')
                exit()
    # If there is no -m argument at all
    else:
        usealpha = False
        masks = None

    # Load what layers the user wants to be involved in the k-Means clustering
    # "layers" defaults to bgr, so that clusters will be ordered along the blue channel
    layers = subarg(CLFLAGS['dims'], 'bgr')[0]
    # layers = re.compile("[^a-zA-Z]").sub('', layers)  # remove any separators

    # Load edtmap (only if the user wanted to use it)
    edtfile = None
    if LAYERIDS['edt'] in layers:  # if there is a switch after --dims for an EDT distance map
        if argexist(CLFLAGS['edt'],
                    False):  # if the user wanted to provide one or more files to load as EDT distance map(s)
            edtfile = subarg(CLFLAGS['edt'])
            if edtfile == '':  # but they have failed to do so
                edtfile = None
                # Running two iterations without a mask deserves a warning.
                if masks is not None:
                    print 'WARNING: EDT distance map was not specified. Two iterations will be performed.'
                else:
                    print ("WARNING: Neither an EDT distance map, nor a mask was specified. "
                           "Please consider removing 'd' from the --dim option. Two iterations will be performed.")
            else:  # if one or more edtfiles have been specified
                if len(edtfile) == len(imglist):  # and it corresponds well to the image files
                    pass  # there is nothing to do here
                else:
                    if len(edtfile) == 1:
                        print 'WARNING: The same EDT distance map will be used for all images.'
                        edtfile = [edtfile[0] for _ in imglist]
                    else:
                        print ('ERROR: The number of EDT distance maps and the number of images are not equal. '
                               'Please amend it. The program now terminates.')
                        exit()
        else:  # if the user hasn't provided any EDT distance maps
            # edtfile is already set to None
            # Running two iterations without a mask deserves a warning.
            if masks is not None:
                print 'WARNING: EDT distance map was not specified. Two iterations will be performed.'
            else:
                print ("WARNING: Neither an EDT distance map, nor a mask was specified. "
                       "Please consider removing 'd' from the --dim option. Two iterations will be performed.")

    # Load parameter settings for the sliding frame
    n_clusters = int(subarg(CLFLAGS['clusters'], 3)[0])
    framesize = subarg(CLFLAGS['frame'], 'auto')
    stepsize = subarg(CLFLAGS['step'], 'auto')

    # Terminate application if no files are selected
    if len(imglist) == 0:
        print ('RESULT: No files were found with the current setting. The program terminates. '
               'Please change the search path or the inclusion/exclusion criteria and try again.')
        exit()

    # Summarize task and insist on user confirmation before proceeding
    line = "\n{:d} files will be processed (excluding {} and {} files):".format(len(imglist), SEGTAG, PROBTAG)
    print line
    print '=' * len(line) + '\n'
    print "\n".join(imglist)
    if masks is not None:
        line = '\n {:d} selected mask(s):'.format(len(masks))
        print line
        print '=' * len(line) + '\n'
        print '\n'.join(masks)
    else:
        print '\n No mask selected.'

    print "\nWould you like to start the process? [yes/no]: "
    if not confirmed_to_proceed():
        exit()

    # Process images
    err = 0
    i = 0
    for imfile in imglist:
        if masks is None:
            maskfile = None
        else:
            maskfile = masks[i]
        if edtfile is None:
            edtf = None
        else:
            edtf = edtfile[i]

        err = err + img_proc(imfile, maskfile, usealpha, layers, edtf,
                             n_clusters, framesize, stepsize)
        i = i + 1

    if err == 0:
        print 'RESULT: The process has been completed successfully.'
    else:
        print 'RESULT: Errors occurred during the process. Some file(s) might have not been created.'


# Main program execution starts here
if __name__ == "__main__":
    # If there are any user-set command-line arguments, do the job.
    if len(sys.argv) > 1:
        main()
    # If there is none, display manual page.
    else:
        print ("""
        The regional-kmeans utility performs repeated k-Means segmentation on images using a sliding-frame approach.
        
        As the sliding frames overlap, pixels are classified multiple times, hence a probabilistic map for each 
        cluster can be generated. Pixels on the borders are sampled fewer times, so make sure that the ROI is in the 
        middle of the input image. 
        
        The output can be either the probability maps of each cluster or a multi-segmented image based on maximum 
        likelihood. 

        Usage: ./regional-kmeans.py [input] [-m] [mask] [options]

        Input options:
            [input]                 image file
                                    for folders, all files in the folder will be processed
                                    leave it empty for the current working folder
            -r                      include subfolders (only if [input] is a folder)
                                    default: No
            --include "patterns,"   use Unix-shell like wildcards to include files (use quotes!)
                                    default: "*.tif,*.tiff"
            --exclude "patterns,"   use Unix-shell like wildcards to exclude files (use quotes!)
                                    defaults to None
        
        Mask options:
            -m [maskfile]           use mask (from file) to exclude background pixels
                                    if [maskfile] is not specified:
                                        RGBA images: a binarised alpha channel will be used
                                        non-RGBA images: a file with identical name and _mask tag will be searched for
        
        Further options:    
            -c <number>             number of segmentation clusters (default: 3) (integer)
            -f <height,width>       sliding frame size in pixels (default: "auto") (both integers)
            -s <dy,dx>              step size (along height, along width) (default: "auto") (both integers)
            --dim <rgb|hsv|LAB|d>   dimensions to be included (mix & match)
                                    (red, green, blue, hue, saturation, value, Lightness, LabA, LabB, distance)
                                    (distance: uses class 0 in second iteration if --edt is not specified)  
            --edt <distance map>    specify distance map file (must have identical dimensions)
            --cpu <num threads>     number of parallel processing units
                                    for single-CPU processing, exclude -p entirely
        
        Output options:
            --no-mlth               suppress the default output: multi-level thresholded image(s)
            --probs                 output probability maps (tag: _class00)
            -b                      save the sampling map(s) (tag: _smap)
            -v                      include parameter settings in the output file names
        """)

# Developer notes
    # open and save numpy array with all classifications (cosider file system limitations)
    # add regex support for the inclusion/exclusion filters
    # treat settings (command line argumets) as an object and extract properties by a method
    # write to error and output buffers separately
    # support for NIfTI and 3D images would be fine
    # parallel computing is not yet supported
    # adjustable runtime constants would be better (there might be only 2)
    # review runtime messages, especially their absence (next file, edt switch, etc)
    # optimize calculations
    # prevent first iteration override (this awaits testing)
    # --edt only works with quotes

# Decisions
    # Despite the larger file size, the TIFF format was chosen over PNG for floating-point type support.
    # The PIL Image Library was chosen over tifffile for multiple file-type support
    # A sampling map is always given for an individual image, not for a session, for later identification