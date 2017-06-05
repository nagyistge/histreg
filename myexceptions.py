#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python


# Breakdown of automatic sliding-frame size determination in regional-kmeans.py
class FrameError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(FrameError, self).__init__(message)


# Invalid output dimension specification in img-to-nifti.py
class FormatError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(FormatError, self).__init__(message)


# Invalid output dimension specification in img-to-nifti.py
class ImgFormatError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(ImgFormatError, self).__init__(message)


# Nothing to do: no image was selected for futher processing.
class NothingToDo(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(NothingToDo, self).__init__(message)


# Item counts don't match
class CountError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(CountError, self).__init__(message)


class AffineError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(AffineError, self).__init__(message)
