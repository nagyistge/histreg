#!/Users/inhuszar/MND_HistReg/MND_HistReg_Python/bin/python

import sys

# The argexist subroutine returns either True or False after checking whether the given
# argument is in the list of the specified command-line arguments. The result will be False
# if the argument takes an obligate subargument that is missing.
def argexist(argv, subarg=False):
    if argv in sys.argv:
        if subarg:
            if len(sys.argv)-1 > sys.argv.index(argv):
                if not sys.argv[sys.argv.index(argv)+1].startswith('-'):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return True
    else:
        return False


# The subarg subroutine check whether the given argument has been specified and returns its subargument.
# If the subargument is missing, a preset default value will be returned.
# Note that the return value is a list.
def subarg(argv, default_value="", splitat=','):
    if argexist(argv, subarg=True):
        arglist = []
        for arg in sys.argv[sys.argv.index(argv)+1:]:
            if not arg.startswith('-'):
                for p in arg.split(splitat):
                    arglist.append(p)
            else:
                break
        return arglist
    else:
        return str(default_value).split(splitat)


# The confirmed_to_proceed forces user input for continuing the execution of the program.
def confirmed_to_proceed(forceanswer=True):

    no = set(['no', 'n'])
    if forceanswer:
        yes = set(['yes', 'y', 'ye'])
    else:
        yes = set(['yes', 'y', 'ye', ''])

    choice = raw_input().lower()
    if choice in yes:
        return True
    elif choice in no:
        return False
    else:
        sys.stdout.write("Please respond with 'yes' or 'no': ")
        confirmed_to_proceed()