#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

import sys


def argexist(argv, subarg=False):
    try:
        arg_idx = sys.argv.index(argv)

        # If the argument takes a sub-argument whose existence needs to be validated:
        if subarg:
            if (len(sys.argv) - 1 > arg_idx) & (sys.argv[arg_idx + 1][0] != '-'):
                return True
            else:
                sys.stderr.write("ERROR: Incorrect value for {} argument.\n".format(argv))
                raise AssertionError("Command-line arguments incorrectly set.\n")
        else:
            return True
    except ValueError as exp:
        return False

    return False


def subarg(argv, default_value=""):
    if argexist(argv, subarg=True):
        return sys.argv[sys.argv.index(argv)+1]
    else:
        return default_value

def confirmed_to_proceed():
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    choice = raw_input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no': ")
        confirmed_to_proceed()


def main():
    import matplotlib.pyplot as plt
    import cv2
    import os
    from sklearn.cluster import KMeans
    import numpy as np
    from timeit import default_timer as timer
    import time
    import datetime
    import tifffile as tiff

    ## Specify path
    #default_path = "/Volumes/INH_1TB/MND_HistReg_Scratch/NP140-14_dissection_photos/Cropped/Blocks/"
    default_path = os.getcwd()
    try:
        FPATH = subarg('-p', default_path).split(',')
    except AssertionError as exp:
        print("Fatal error. Program terminates.\n", exp)
        exit()

    ## Specify extensions
    try:
        extlist = subarg('-e', "tif,tiff").split(',')
    except AssertionError as exp:
        print("Fatal error. Program terminates.\n", exp)
        exit()
    EXTENSIONS = tuple(".{}".format(extlist))

    ## Specify the number of clusters
    try:
        N_CLUSTERS = int(subarg('-c', 2))
    except AssertionError as exp:
        print("Fatal error. Program terminates.\n", exp)
        exit()

    ## Specify output: masks only (default) / masks and images / images only
    MASKTAG = '_mask.tif'
    SEGTAG = '_seg.tif'
    savemask, applymask = True, False
    if (argexist('-a')):
        if (N_CLUSTERS == 2):
            applymask = True
            if not argexist('-m'):
                savemask = False
        else:
            print "N_CLUSTERS = {}".format(N_CLUSTERS)
            sys.stderr.write("WARNING: Segmentation mask not applicable if not binary. Output will be mask-only.\n")

    imglist = []
    for fp in FPATH:
        for path, subdirs, files in os.walk(fp, topdown=True):
            if not argexist('-r'):
                subdirs[:] = []
            for file in files:
                if (file[0] != '.') & (file.lower().endswith(EXTENSIONS)) & \
                        (not file.lower().endswith(MASKTAG)) & \
                        (not file.lower().endswith(SEGTAG)):
                    imglist.append(os.path.join(path, file))

    print "\nFiles to be processed (excluding _seg and _mask files):"
    print "=======================================================\n"
    print "\n".join(imglist)
    print "\nWould you like to start the process? [yes/no]: "
    if confirmed_to_proceed() == False:
        exit()

    # Loop through the list of images
    start = timer()
    t = datetime.datetime.fromtimestamp(time.time())
    print "Process started at {}".format(t.strftime('%Y-%m-%d %H:%M:%S'))

    for img in imglist:
        # Load image
        print "Reading {}...".format(img)
        im = np.asarray(tiff.imread(img))
        h,w = im.shape[:2]

        # Extract hue channel
        ch = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:,:,0]

        # Segment image using KMeans clustering
        ch = np.reshape(ch, (-1,1))
        kmeans = KMeans(n_clusters=N_CLUSTERS).fit(ch)
        mask = np.asarray(kmeans.labels_, dtype=np.int8).reshape((h,w), order='C')

        # Invert mask if the top-left pixel is the brightest (permissive).
        if mask[0,0] == np.max(mask):
            mask = np.max(mask) - mask

        # Normalise mask labels so that they can be seen in a regular image viewer
        mask = mask / np.max(mask) * 255
        #mask[mask > 0] = 1 ### Decision: don't automatically binarize masks.

        # Create output
        if savemask:
            tiff.imsave(img[:-4] + MASKTAG, mask[np.newaxis, np.newaxis, np.newaxis, :, :].astype(np.uint8))
            print "Mask file saved as {}.".format(img.split(os.sep)[-1][:-4] + MASKTAG)
        if applymask:
            seg = np.copy(im)
            seg[mask==0] = 0
            seg = np.rollaxis(seg, -1, 0).astype(np.uint8)
            tiff.imsave(img[:-4] + SEGTAG, seg[np.newaxis, np.newaxis, :, :, :])
            print "Segmented image saved as {}.".format(img.split(os.sep)[-1][:-4] + SEGTAG)

    end = timer()
    t = datetime.datetime.fromtimestamp(time.time())
    print "\nSegmentation was completed at {} ({:.1f} seconds)".format(t.strftime('%Y-%m-%d %H:%M:%S'), end - start)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        print """
        The segment-by-hue utility performs k-Means segmentation on digital photographs using the values of the Hue colour channel.
        The output can be a label mask, a segmented image or both.
        
        Usage: ./segment-by-hue.py [options]
        
        Options:
            -p <path1,path2...>     root path of image files (add multiple paths as comma-separated list)
                                    the default is the current working directory
            -r                      process files in subdirectories
            -e <ext1,ext2...>       file extensions to be included (as comma-separated list)
            
            -c <number>             number of segmentation clusters (default: 2)
            
            -a                      apply segmentation mask (binary mask is needed: use -c 2)
            -m                      save mask file(s) (only needed if -a is used)
                    
        """