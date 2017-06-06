#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-Jun-05
# Only single file processing is implemented at the moment.

import numpy as np
import tifffile as tiff
from sklearn.cluster import KMeans
from mpi4py import MPI
import os
import psutil
from args import *


class JobDescriptorObj:
    def __init__(self, img, bgdp, roi_centres, n_features, n_clusters,
                 frame_height, frame_width, weights):
        self.img = img
        self.bgdp = bgdp
        self.roi_centres = roi_centres
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.weights = weights


# This listing is only provisional.
CLFLAGS = {'subdir': '-r',
           'exclude': '--exclude',
           'include': '--include',
           'mask': '-m',
           'dp': '--bgdp',
           'clusters': '-c',
           'frame': '-f',
           'dims': '--dims',
           'random': '--random',
           'edt': '--edt',
           'cpu': '--cpu',
           'smap': '--smap',
           'mlth': '--mlth',
           'probs': '--probs',
           'verbose': '-v',
           'weight': '-w'}


def shift_image(img, (x, y), inverse=False):
    # The image must be 2D
    if not inverse:
        return np.roll(np.roll(img, y, 0), x, 1)
    else:
        return np.roll(np.roll(img, -y, 0), -x, 1)


def run_kmeans(job):
    fh, fw = job.frame_height, job.frame_width
    h, w, ch = job.img.shape
    subtotal = job.roi_centres[:,0].size

    # Create stack for classification results
    # (layers: number of clusters + background)
    results = np.zeros((h, w, job.n_clusters + 1))
    smap = np.zeros((h, w))

    i = 0
    # Loop through the ROIs
    for y, x in job.roi_centres:
        # Define frame
        top = max(y - fh/2, 0)
        bottom = min(y + fh/2, h)
        left = max(x - fw / 2, 0)
        right = min(x + fw / 2, w)
        frame = np.array(job.img[top:bottom, left:right, :])
        frame_mask = np.array(job.bgdp[top:bottom, left:right])

        # Pool data from the frame by discarding background and dark pixels
        data_all = frame.reshape((-1, job.n_features), order='F')
        data = data_all[(frame_mask == 0).reshape(-1, order='F'), :]
        # Prevent ValueError, when the frame is entirely in the mask area
        # Count as detection for mask class
        # (added on 5th June, 2017)
        if data.size < job.n_clusters:
            results[top:bottom, left:right, 0] = \
                results[top:bottom, left:right, 0] + 1
            continue
        data = data - np.mean(data, axis=0)
        data = data / np.var(data, axis=0)

        # Change invalid values (div by zero) to 0
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        data[np.isneginf(data)] = 0

        # Perform k-Means clustering (prepare for hitting the bug)
        try:
            # Set np.float64 to avoid a bug in the k-means implementation.
            data = data.astype(np.float64)
            kmeans = KMeans(n_clusters=job.n_clusters, random_state=1).fit(data)

            # Reintroduce zero-pixels and reshape labels array
            # The axis swap is a trick to comply with earlier Fortran-reshapes.
            labels = np.rollaxis(np.zeros_like(frame_mask), 1, 0)
            frame_mask = np.rollaxis(frame_mask, 1, 0)
            labels[frame_mask == 0] = 1 + kmeans.labels_  # 0 is for background
            labels = np.rollaxis(labels, 1, 0)

            # Standardise labels
            # 1: dark pixels, 2: GM, 3: WM
            # order labels by the first feature values
            order = 1 + np.argsort(kmeans.cluster_centers_[:, 0])
            if not all(order[i] <= order[i + 1] for i in
                       xrange(len(order) - 1)):
                tmp = np.copy(labels)
                for label in range(1, job.n_clusters, 1):
                    labels[tmp == label] = 1 + np.where(order == label)[0][0]

        # If the classifier hits the bug, set all labels to -1,
        #  so they won't be counted.
        except IndexError:
            print 'IndexError'
            err = err + 1
            continue

        # Stack classification results
        for clust in range(job.n_clusters + 1):
            inc = np.zeros_like(labels)
            inc[np.where(labels == clust)] = 1
            results[top:bottom, left:right, clust] = \
                results[top:bottom, left:right, clust] + inc

        # Update sampling map (if necessary)
        smap[top:bottom, left:right] = smap[top:bottom, left:right] + 1

        # Report progress
        i = i + 1
        print '{}. process: {}/{}'.format(rank, i, subtotal)

    return results, smap


def main_proc(comm):
    print 'Main process running.'
    print 'Number of parallel processes: {}'.format(n_workers)

    # Load original image
    imfile = ''
    if argexist(sys.argv[0], True):
        imfile = subarg(sys.argv[0])[0]
    # The following two lines are here just for the developer's convenience
    if imfile == '':
        imfile = '../../edge-rkms-experimental/NP140-14 (97).tif'
    if not os.path.isfile(imfile):
        print "ERROR: The image could not be loaded from '{}'".format(imfile)
        comm.Abort()
        exit()
    fpath, fname = os.path.split(imfile)
    img = tiff.imread(imfile)
    h, w, ch = img.shape
    # FIXME: this is a crude way of getting rid o the alpha channel
    if ch > 3:
        img = img[:,:,:3]
        ch = 3

    # Load the combined background and dark pixels mask
    bgdpfile = subarg(CLFLAGS['dp'])[0]
    # The following two lines are here just for the developer's convenience
    if bgdpfile == '':
        bgdpfile = os.path.join(fpath, 'bgdp.tif')
    if not os.path.isfile(bgdpfile):
        print "ERROR: The background and dark pixel map could not be loaded " \
              "from '{}'".format(bgdpfile)
        comm.Abort()
        exit()
    bgdp = tiff.imread(bgdpfile)

    # Validate that the two images are compatible:
    if bgdp.shape != img.shape[:2]:
        print 'The mask and the main images have different shapes.'
        comm.Abort()
        exit()

    if argexist(CLFLAGS['edt'], True):
        edtfile = subarg(CLFLAGS['edt'])[0]
        if os.path.isfile(edtfile):
            edt = tiff.imread(edtfile)
            if edt.shape != img.shape[:2]:
                print 'EDT filed had incompatible shape.'
                comm.Abort()
                exit()
            else:
                img = np.dstack((img, edt[:, :, np.newaxis]))
                h, w, ch = img.shape
        else:
            print 'EDT file could not be opened from {}.'.format(edtfile)
            comm.Abort()
            exit()

    # Find edges and define edge pixels as ROI centres
    step = 1
    edges = np.zeros_like(bgdp)
    for x in step*np.asarray([-1, 0, 1], dtype=np.int8):
        for y in step*np.asarray([-1, 0, 1], dtype=np.int8):
            if (x, y) != (0, 0):
                diff = bgdp - shift_image(bgdp, (x, y))
                diff[diff != 0] = 1
                edges = np.logical_or(edges, shift_image(diff, (x, y),
                                                         inverse=True)).astype(
                                                                        np.int8)

    roi_centres = np.vstack(np.where(edges != 0)).T
    total = roi_centres[:, 0].size

    if argexist(CLFLAGS['random'], True):
        try:
            r = int(subarg(CLFLAGS['random'], 1000)[0])
        except:
            print 'Invalid {} argument.'.format(CLFLAGS['random'])
            comm.Abort()
            exit()
        idx = np.random.randint(0, total, r)
        roi_centres = roi_centres[idx, :]
        total = roi_centres[:, 0].size

    else:
        # Reduce the number of ROIs
        for x in step*np.asarray([-1, 0, 1], dtype=np.int8):
            for y in step*np.asarray([-1, 0, 1], dtype=np.int8):
                if (x, y) != (0, 0):
                    diff = shift_image(edges - shift_image(edges, (x, y)), (x, y),
                                       inverse=True)
                    edges[diff == 0] = 0

        roi_centres = np.vstack(np.where(edges != 0)).T
        total = roi_centres[:,0].size

    print 'Number of ROIs:', total

    # Set constants for frame size and counts of clusters and features
    n_features = ch
    print 'Features: ', n_features
    # FIXME: no exception handling
    n_clusters = int(subarg(CLFLAGS['clusters'], 3)[0])
    fh, fw = np.array(subarg(CLFLAGS['frame'], "301,301")).astype(np.int32)
    weights = np.array(subarg(CLFLAGS['weight'], ','.join(['1' for i in
                                                           range(ch)])), dtype=np.float64)
    if weights.size != ch:
        print 'Weight vector had incompatible size.'
        comm.Abort()
        exit()
    else:

        #wsum = np.sum(weights)
        weights = weights / np.sum(weights)
        # FIXME: I ONLY HOPE THAT THIS IS CORRECT:
        img = np.multiply(img, weights[np.newaxis, np.newaxis, :])

    # Create JobDescriptorObj and pass it to the parallel processes
    print 'Distributing the job among {} workers...'.format(n_workers)
    comm.Barrier()
    jobs = []
    unit = total / n_workers
    for worker in range(0, n_workers):
        start = worker * unit
        end = start + unit
        if worker == n_workers - 1:
            end = total
        jobs.append(JobDescriptorObj(img, bgdp, roi_centres[start:end,:],
                                     n_features, n_clusters, fh, fw, weights))
    for worker in range(len(jobs)):
        if worker != 0:
            comm.send(jobs[worker], dest=worker, tag=11)

    n_ROIs = np.array(jobs[0].roi_centres[:, 0].size, 'd')
    s = np.zeros_like(n_ROIs)
    comm.Reduce([n_ROIs, MPI.DOUBLE], [s, MPI.DOUBLE], op=MPI.SUM, root=0)
    if s == total:
        print 'Distribution of jobs was successful.'
    else:
        print 'Some ROIs were lost while distributing jobs.'
        comm.Abort()
        exit()

    # Perform k-means classification in the main process
    print 'Main process is working...'
    results, smap = run_kmeans(jobs[0])

    # Pooling results from all workers
    comm.Barrier()
    print 'Pooling results from all processess...'
    results_all = np.zeros_like(results)
    comm.Reduce([results, MPI.DOUBLE], [results_all, MPI.DOUBLE], op=MPI.SUM,
                root=0)
    comm.Barrier()
    smap_all = np.zeros_like(smap)
    comm.Reduce([smap, MPI.DOUBLE], [smap_all, MPI.DOUBLE], op=MPI.SUM,
                root=0)

    # Save results table as numpy array
    np.save('clustertable', results_all)

    # Generate output
    sums = np.sum(results_all, axis=2)
    probs = np.asarray(results_all, dtype=np.float32) / sums[:, :, np.newaxis]
    segmentation = np.argmax(probs, axis=2)
    checksum = np.sum(probs, axis=2)
    print "Checksum (=1 =1): {}, {}".format(np.min(checksum), np.max(checksum))

    # Save segmentation result
    if argexist(CLFLAGS['mlth']):
        fn = os.path.join(fpath, fname[:-4]) + '_mlth.tif'
        tiff.imsave(fn, segmentation.astype(np.uint8))
        print 'SAVED:', fn

    # Save probability maps
    if argexist(CLFLAGS['probs']):
        for clss in range(n_clusters + 1):
            fn = os.path.join(fpath, fname[:-4]) + '_class{0:02d}.tif'.format(clss)
            tiff.imsave(fn, probs[:, :, clss].astype(np.float32))
            print 'SAVED:', fn

    # Save sampling map
    if argexist(CLFLAGS['smap']):
        fn = os.path.join(fpath, fname[:-4]) + '_smap.tif'
        tiff.imsave(fn, smap_all.astype(np.uint32))
        print 'SAVED:', fn

    return 0


def parallel_proc(comm):
    print 'Process number {} ready.'.format(rank)
    # Wait for image descriptor object to arrive
    comm.Barrier()
    job = comm.recv(source=0, tag=11)

    # Report how many ROIs this process has received
    n_ROIs = np.array(job.roi_centres[:,0].size, 'd')
    comm.Reduce([n_ROIs, MPI.DOUBLE], None, op=MPI.SUM, root=0)

    # Do the job and pool the results individually
    print 'Process {} is working...'.format(rank)
    results, smap = run_kmeans(job)

    # Send the results as a numpy array
    comm.Barrier()
    comm.Reduce([results, MPI.DOUBLE], None, op=MPI.SUM, root=0)
    comm.Barrier()
    comm.Reduce([smap, MPI.DOUBLE], None, op=MPI.SUM, root=0)

    # Close instance
    exit()


# Main program execution starts here
p = psutil.Process(os.getppid())

# FIXME: add exception handling
if str(p.name()).lower() != 'orterun':
    n_jobs = int(subarg(CLFLAGS['cpu'], 1)[0])
    if n_jobs > 1:
        os.system('mpirun -n {} python '.format(n_jobs) + ' '.join(sys.argv))
        exit()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

if rank == 0:
    main_proc(comm)
elif rank != 0:
    parallel_proc(comm)