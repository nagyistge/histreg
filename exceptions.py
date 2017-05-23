#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python


# Breakdown of automatic sliding-frame size determination
class FrameError(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super(FrameError, self).__init__(message)

        # Now for your custom code...
        self.errors = errors