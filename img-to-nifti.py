#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

# 2017-May-29
# Main development branch. Major update to the syntax of the --dims argument.

import numpy as np
import nibabel
from args import *
from myexceptions import *
from PIL import Image
import os
import fnmatch
from itertools import count


# Define command-line argument flags
CLFLAGS = {'subdir': '-r',
           'exclude': '--exclude',
           'include': '--include',
           'uint': '--uint',
           'dims': '--dims',
           'vsizes': '--vsizes',
           'sform': '--sform',
           'qform': '--qform',
           'concat': '--concat',
           'out': '--out'}

WCARDS = {'width': 'w', 'height': 'h', 'layers': 'l', 'number': 'n'}

# Define greyscale modes (in accordance with PIL Image modes)
GMODES = {'1', 'L', 'P', 'I', 'F'}

# Define inclusion/exclusion filters
INC = subarg(CLFLAGS['include'], '*.tif,*.tiff')  # default to TIFF files
EXC = subarg(CLFLAGS['exclude'])
if argexist(CLFLAGS['exclude'], subarg=True):
    # if only --exclude is specified, automatically include all files
    if not argexist(CLFLAGS['include'], subarg=True):
        INC = '*'  # note that *.* would exclude files without extension


class FormatDescriptorObj:
    # Class defaults
    _shape = [1, 1, 1, 1]
    _shc = [1, 1, 1, 1]
    _qform = np.eye(4)
    _sform = None
    _dtype = np.float64
    _axisorder = [0, 1, 2]

    # Constructor
    def __init__(self, shape=_shape, shc=_shc, qform=_qform, sform=_sform,
                 axisorder=_axisorder, dtype=_dtype):
        self.shape = shape
        self.shc = shc    # shape coefficient
        self.qform = qform
        self.sform = sform
        self.axisorder = axisorder
        self.dtype = dtype

    # Setter
    def setShape(self, index, shape, shc):
        self.shape[index] = shape
        self.shc[index] = shc

    def extract(self):
        return self.shape, self.signs, self.dtype


# Collect paths to all files that should be processed
def parse_images():
    imglist = []
    inpath = subarg(sys.argv[0], default_value=os.getcwd())
    for fp in inpath:
        # If the current path is a directory
        if os.path.isdir(fp):
            if os.path.split(fp)[-1] == '.':
                fp = os.getcwd()
            for path, subdirs, files in os.walk(fp, topdown=True):
                # Exclude subdirectories unless specified
                if not argexist('-r'):
                    subdirs[:] = []
                for file in files:
                    # Exclude hidden files, files with output tags,
                    # apply user-set inclusion/exclusion filters
                    if (file[0] != '.') & \
                            (not file.lower().endswith('.nii')) & \
                            (not file.lower().endswith('.nii.gz')) & \
                            (any(fnmatch.fnmatch(file, p) for p in INC)) & \
                            (not any(fnmatch.fnmatch(file, p) for p in EXC)):
                        imglist.append(os.path.join(path, file))

        # If the current path points at a file
        elif os.path.isfile(fp):
            fn = os.path.split(fp)[-1]
            # Exclude hidden files, files with output tags,
            # apply user-set inclusion/exclusion filters
            if (fn[0] != '.') & \
                    (not fn.lower().endswith('.nii')) & \
                    (not fn.lower().endswith('.nii.gz')) & \
                    (any(fnmatch.fnmatch(fn, p) for p in INC)) & \
                    (not any(fnmatch.fnmatch(fn, p) for p in EXC)):
                # Add local files with their full path,
                # so that duplicates can be removed
                if os.path.split(fp)[0] == '':
                    fp = os.path.join(os.getcwd(), fp)
                imglist.append(fp)
        else:
            print 'WARNING: Path {} could not be recognised.'.format(fp)

    # Remove possible duplicates
    imglist = list(set(imglist))

    if len(imglist) == 0:
        raise NothingToDo('No images have been selected. Please try again.')

    return imglist


def parse_format():
    # Create format descriptor object
    imgformat = FormatDescriptorObj()

    # Read user-set dimension arguments from the command line
    dims = subarg(CLFLAGS['dims'], '[w,h,l,1]')

    # Check brackets (as leading minus signs would be confused with arguments)
    if (dims[0][0] != '[') | (dims[-1][-1] != ']'):
        raise FormatError('ERROR: Square brackets are necessary for the format '
                          'specification that follows the {} argument.'.format(
                          CLFLAGS['dims']))
    # Strip brackets
    dims[0] = dims[0][1:]
    dims[-1] = dims[-1][:-1]

    # Assertion: three spatial or a total four parameters were specified
    if len(dims) == 3:
        dims.append('1')    # being user-friendly: add obvious time dimension
    if len(dims) != 4:
        raise FormatError('ERROR: Invalid output format. Please specify exactly'
                          ' 4 parameters.')

    # Assertion: all spatial wildcards must appear once and exactly once
    for key in [k for k in WCARDS.keys() if k != 'number']:
        if np.flatnonzero(np.array([WCARDS[key] in dim for dim in dims[:-1]]))\
                .size != 1:
            raise FormatError("ERROR: '{}' wildcard must appear exactly once."
                              .format(WCARDS[key]))

    # Assertion: time dim can only have the concat wildcard, only when needed
    if argexist(CLFLAGS['concat']) != (WCARDS['number'] in dims[-1]):
        raise FormatError("ERROR: The {} argument must and the wildcard '{}' "
                          "in the time dimension must be set jointly.".format(
                          CLFLAGS['concat'], WCARDS['number']))

    # Assertion: no alien wildcard in the time dimension
    if (not str(dims[-1][-1]).isdigit()) | \
        (str(dims[-1][-1]).lower() == WCARDS['number']):
        raise FormatError('ERROR: Invalid wildcard in the time dimension.')

    # Translate format code to FormatDescriptorObj's shape and signs
    axisorder = []
    for dim in list(dims[:-1]):
        # Store index
        index = dims.index(dim)

        # Determine which image dimension is specified
        if dim[-1] == WCARDS['width']:
            axisorder.append(0)             # 0 is for x
        elif dim[-1] == WCARDS['height']:
            axisorder.append(1)             # 1 is for y
        elif dim[-1] == WCARDS['layers']:
            axisorder.append(2)             # 2 is for z

        # Discard the wildcard character from the end
        shape = str(dim[-1])
        dim = dim[:-1]

        # Extract signed coefficient
        try:
            if (dim == '+') | (dim == ''):
                shc = 1
            elif dim == '-':
                shc = -1
            else:
                shc = int(dim)
        except ValueError:
            raise FormatError('ERROR: Unrecognised coefficient in {}.'
                              .format(dims[index]))

        # Store shape and sign in the FormatDescriptorObj
        imgformat.setShape(index, shape, shc)

    # Store axisorder in the FormatDescriptorObj
    imgformat.axisorder[:] = axisorder


    # Read data type argument from the command line
    if argexist(CLFLAGS['uint']):
        imgformat.dtype = np.uint8
    else:
        imgformat.dtype = np.float64

    # Verify the integrity of user-set voxel sizes
    sizes = subarg(CLFLAGS['vsizes'], '1,1,1,1')
    if len(sizes) != 4:
        raise FormatError('ERROR: Invalid voxel size description. Please '
                          'specify exactly 4 parameters.')
    try:
        sizes = np.array([float(s) for s in sizes])
        if np.any(sizes <= 0):
            raise FormatError('ERROR: All voxel sizes must be positive. ')
        # 1,1,1,1 default has already been initialised by FormatDescriptorObj()
        elif np.any(sizes != np.array([1, 1, 1, 1])):
            imgformat.qform = np.diag(sizes)
        else:
            pass

    except SyntaxError:
        raise FormatError('ERROR: Invalid value encountered in voxel size '
                          'descriptors.')

    # Parse input for affine matrices
    qformlist = [matfile for matfile in subarg(CLFLAGS['qform'])
                 if os.path.isfile(matfile)]
    sformlist = [matfile for matfile in subarg(CLFLAGS['sform'])
                 if os.path.isfile(matfile)]
    if (len(sformlist) * len(qformlist) > 0) & \
            (len(sformlist) != len(qformlist)):
        raise CountError('ERROR: The numbers of sform and qform matrices '
                         'are not equal.')

    # Parse output settings
    outfilelist = subarg(CLFLAGS['out'])
    if outfilelist == ['']:
        if argexist(CLFLAGS['concat']):
            print ('ERROR: You must specify an output file name if {} is set.'
                   .format(CLFLAGS['concat']))
            exit()
        else:
            outfilelist = []
    else:
        # Remove potential duplicates to prevent overwriting
        # FIXME: this only works if there are no corresponding absolute and relative pairs of paths for the same file
        outfilelist = set(outfilelist)

    return imgformat, qformlist, sformlist, outfilelist


def summarize(imglist, imgformat, qformlist, sformlist, outfilelist):
    # If qform/sform is specified, check if there is one for each image, or
    # set the general qform/sform matrix of the format object to the one given.
    # For any other case, flag an error and terminate the program.
    if argexist(CLFLAGS['qform'], True):
        if len(qformlist) != len(imglist):
            if len(qformlist) == 1:
                print 'The same qform matrix will be used for all images.'
                imgformat.qform = np.loadtxt(qformlist[0])
                qformlist = []  # erase list if there is only one item
            else:
                raise CountError('The numbers of qform matrices and input'
                                 'images are not equal.')

    if argexist(CLFLAGS['sform'], True):
        if len(sformlist) != len(imglist):
            if len(sformlist) == 1:
                print 'The same qform matrix will be used for all images.'
                imgformat.sform = np.loadtxt(sformlist[0])
                sformlist = []  # erase list if there is only one item
            else:
                raise CountError('The numbers of sform matrices and input'
                                 'images are not equal.')

    if outfilelist:
        if not argexist(CLFLAGS['concat']):
            if len(outfilelist) != len(imglist):
                if len(outfilelist) == 1:
                    print 'Output filename will be tagged with an ordinal.'
                    outfilelist[:] = \
                        [os.path.join(os.path.split(imfile)[0],
                                      '.'.join(os.path.split(imfile)[-1]
                                               .split('.')[:-1]) +
                         '_{}.nii.gz'.format(imglist.index(imfile)))
                         for imfile in imglist]
                else:
                    raise CountError('The number of input and output files does'
                                     'not match.')
            else:
                pass  # Nothing to do, everything is fine.
        else:
            print ('ERROR: Multiple output files specified when the {} flag is '
                   'set. The program now terminates.'.format(CLFLAGS['concat']))
            exit()
    else:
        outfilelist[:] = \
            [os.path.join(os.path.split(imfile)[0],
                          '.'.join(os.path.split(imfile)[-1].split('.')[:-1]) +
                          '.nii.gz') for imfile in imglist]

    # Display the files awaiting conversion (source -> destination)
    headline = "\n{:d} file(s) will be processed (source -> destination) " \
           "(excluding NIfTI files):".format(len(imglist))
    print headline
    print '=' * len(headline)
    print "\n".join(['  '+str(count(1).next())+'.  '+line[0] + ' -> ' + line[1] for line in
                     zip(imglist, outfilelist)])

    # Display list of qform files
    if qformlist:  # shorthand for "qformlist != []"
        line = '\nSelected qform matrices ({:d}):'.format(len(qformlist))
        print line
        print '=' * len(line)
        print '\n'.join(['  '+str(count(1).next())+'.  '+q for q in qformlist])
    else:
        print '\n   No qform matrix was selected. Identity matrix will be used.'

    # Display list of sform files
    if sformlist:  # shorthand for "sformlist != []"
        line = '\nSelected sform matrices ({:d}):'.format(len(sformlist))
        print line
        print '=' * len(line)
        print '\n'.join(['  '+str(count(1).next()) + '.  ' + s for s in sformlist])
    else:
        print '\n   No sform matrix was selected. Only qform will be used.'


def load_affine_from_list(affine_list, index=0, default=np.eye(4)):
    if affine_list:
        try:
            mat = np.loadtxt(affine_list[index]).astype(np.float64)
            if mat.shape != (4, 4):
                raise AffineError('Affine matrix at {} had invalid shape.'
                                  .format(affine_list[index]))
        except IOError:
            raise AffineError('Affine matrix at {} was not found.'.format(
                affine_list[index]))
        except ValueError:
            raise AffineError('Invalid value in affine matrix at {}.'
                              .format(affine_list[index]))
        return mat
    else:
        return default


def cast_img(img, shape, shc, warning=False):
    newimg = np.copy(img) # This is a 4-dimensional image object
    oldshape = newimg.shape
    newshape = np.array(shape)
    if np.any(np.mod(newshape, oldshape) != 0):
        raise FormatError('shape {} could not be cast into the output shape {}'
                          .format(str(tuple(oldshape)), str(tuple(newshape))))

    # If necessary, broadcast image via repeat along dims from left to right
    cropped = False
    diff_idx = np.where(newshape != oldshape)[0]
    if np.any(diff_idx):  # if there is any difference
        reps = np.floor_divide(newshape, oldshape)
        for dim in diff_idx:
            # if the new shape dimension is an even multiple of the old one,
            # fill by repetition
            if newshape[dim] % oldshape[dim] == 0:
                newimg = np.repeat(newimg, reps[dim], axis=dim)
            # if not, repeat and crop
            else:
                cropped = True
                newimg = np.repeat(newimg, reps[dim] + 1, axis=dim)
                newimg = np.array_split(newimg, [newshape[dim]], axis=dim)[0]

    # Apply the necessary flips (from signs)
    for dim in np.where(np.array(shc) < 0)[0]:
        newimg = np.flip(newimg, axis=dim)

    # Display warning if needed
    if warning & cropped:
        print 'WARNING: The image had to be cropped to fit into the new shape.'

    return newimg


def sortaxes(ndarray, new_order):
    newarr = np.copy(ndarray)
    for passnum in range(len(new_order) - 1, 0, -1):
        for i in range(passnum):
            if new_order[i] > new_order[i + 1]:
                temp = new_order[i]
                new_order[i] = new_order[i + 1]
                new_order[i + 1] = temp
                newarr = np.swapaxes(newarr, i, i + 1)
    return newarr


def reformat(imglist, imgformat, qformlist, sformlist, outfilelist,
             concat=False):
    err = 0

    for imfile in imglist:
        # Open image
        try:
            pilimg = Image.open(imfile)
        except IOError:
            print 'ERROR while reading {}. The image was skipped.'
            err = err + 1
            continue

        # Make sure that the image is greyscale
        if not pilimg.mode in GMODES:
            pilimg = pilimg.convert('L')  # convert colour images to 8-bit grey

        # Set data type
        img = np.array(pilimg, dtype=np.float64)
        if imgformat.dtype != np.float64:
            img = np.array((img - np.min(img)) / np.max(img) * 255,
                           dtype=imgformat.dtype)

        # Reorient image from yxz to xyz
        img = np.swapaxes(img, 0, 1)

        # Convert image into a 4-dimensional object
        if len(img.shape) == 1:
            img = img[:, np.newaxis, np.newaxis, np.newaxis]
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis, np.newaxis]
        elif len(img.shape) == 3:
            img = img[:, :, :, np.newaxis]
        else:
            print ('Input image at {} had invalid shape. '
                   'The image was skipped.'.format(imfile))
            continue

        # Reorient image according to the wildcards
        img = sortaxes(img, imgformat.axisorder)

        # Calculate new dimensions
        newshape = np.abs(imgformat.shc) * np.array(img.shape)
        imgformat.shape = newshape

        # Load qform and sform matrices and check their integrity
        try:
            qmat = load_affine_from_list(qformlist, imglist.index(imfile),
                                         imgformat.qform)
        except AffineError as exc:
            print (exc.message + ' The corresponding image at {} was skipped.'
                   .format(imfile))
            continue
        try:
            smat = load_affine_from_list(sformlist, imglist.index(imfile),
                                         imgformat.sform)
        except AffineError as exc:
            print (exc.message + ' The corresponding image at {} was skipped.'
                   .format(imfile))
            continue

        # Skip image if the determinants have different signs
        if smat is not None:
            if np.linalg.slogdet(qmat)[0] != np.linalg.slogdet(smat)[0]:
                print ('Qform and sform matrix determinants have different '
                       'signs. The corresponding image at {} was skipped.'
                       .format(imfile))
                continue

        # If not meant to be concatenated, save images separately
        if not concat:
            # Create NIfTI header
            hdr = nibabel.Nifti1Header()
            hdr.set_dim_info(slice=2)  # Set Z as slice encoding axis.
            hdr.set_data_shape(newshape)
            hdr.set_xyzt_units(2, 16)  # nifti codes for mm and msec
            hdr.set_qform(qmat, 1, strip_shears=True)
            if smat is not None:
                hdr.set_sform(smat, 2)
            hdr.set_data_dtype(imgformat.dtype)

            # Create, cast and save an instance of nibabel.Nifti1Image
            try:
                newimg = cast_img(img, imgformat.shape, imgformat.shc, True)
            except FormatError as exc:
                print ('Input image at {} with '.format(imfile) + exc.message +
                       ' The image was skipped.')
                continue
            except ImgFormatError as exc:
                print (exc.message + 'The image at {} was skipped.'
                       .format(imfile))
                continue
            nimg = nibabel.Nifti1Image(newimg, hdr.get_best_affine(), hdr)
            try:
                outfile = outfilelist[imglist.index(imfile)]
                nibabel.save(nimg, outfile)
                print 'SAVE: {} was created successfully.'.format(outfile)
            except IOError:
                print 'ERROR: The file at {} was not created.'.format(imfile)

        else:
            pass  # Literally, this is a very hard thing to do, so pass.


def main():
    # Load images
    try:
        imglist = parse_images()
    except NothingToDo as exc:
        print exc.message + ' The program now terminates.'
        exit()

    # Parse format specification
    try:
        imgformat, qformlist, sformlist, outfilelist = parse_format()
    except (FormatError, SyntaxError) as exc:
        print exc.message + ' The program now terminates.'
        exit()

    # Summarize task and insist on user confirmation before proceeding
    try:
        summarize(imglist, imgformat, qformlist, sformlist, outfilelist)
        print "\nWould you like to start the process? [yes/no]: "
        if not confirmed_to_proceed():
            exit()
        print 'RESULTS:'
        print '========'
    except CountError as exc:
        print exc.message + ' The program now terminates.'
        exit()

    # Reformat image(s)
    reformat(imglist, imgformat, qformlist, sformlist, outfilelist,
             concat=argexist(CLFLAGS['concat']))


# Main program execution starts here
if __name__ == "__main__":
    # If there are any user-set command-line arguments, do the job.
    if len(sys.argv) > 1:
        # Check for typos
        if np.any(np.logical_not(np.in1d(np.array([arg for arg in sys.argv[1:]
                if str(arg).startswith('-')]), CLFLAGS.values()))):
            print 'Unrecognised argument.'
        else:
            # Launch
            main()
    # If there is none, display manual page.
    else:
        print ("""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The img-to-nifti utility converts 2D/3D images from ordinary image file 
    types into the NIfTI-1 format designed for MR Neuroimaging. Colour images 
    are transformed to grayscale before conversion.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Supported conversions:
        dimensions: 2D->2D, 2D->3D, n*2D->3D, 3D->3D, n*3D->4D, 2D->4D
        input formats: all formats supported by Python Pillow.

    Output: one or more .nii.gz file(s). 

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    +   Usage: ./img-to-nifti.py [input options] [options] [-out <output>]    +
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Input options:
        [input]                 image file(s) (Unix wildcards allowed in path)
                                folder(s): all files in the folder(s)
                                default: current working folder
        -r                      include subfolders ([input] must be a folder)
                                default: No
        --include "patterns,"   Unix-like file mask for inclusion, (use quotes!)
                                default: "*.tif,*.tiff"
        --exclude "patterns,"   Unix-like file mask for exclusion, (use quotes!)
                                defaults to None

    Conversion options:    
        --uint                  use 8-bit uint values (default: 32-bit float)
        --dims [x,y,z,t]        output dimensions (controls repetitions, flips)
                                wildcards: (+/-h)eight, (+/-w)idth, (+/-l)ayers,
                                           (+/-)number of images
                                default: [w,h,l,1] (input image dimensions)
                                BEWARE: square brackets must be used!
        --vsizes <x,y,z,t>      voxel sizes (in mm and ms)
                                default: <1,1,1,1>
        --qform <qform_basic.mat>     set output affine by qform matrix (mm)
                                default: vsizes, if not: identity matrix
        --sform <sform.mat>     set output affine by sform matrix (mm)
                                default: identity matrix
        --concat                concatenate input images (use wildcard n)

    Output options:
        --out <output>          default: files are saved as input with .nii.gz
                                else: output count must match input count
                                (a single file must be set for --concat)
        """)


        # Developer notes
        # TODO: Add support for xml header input.
        # TODO: Add support for NIfTI header input
        # TODO: General error sum
        # TODO: Concatenation of multiple images into a single NIfTI file