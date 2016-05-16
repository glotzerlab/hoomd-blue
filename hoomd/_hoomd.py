# This file exists to allow the hoomd module to import from the source checkout dir
# for use when building the sphinx documentation.

class Messenger(object):
    def openPython(self):
        pass

    def notice(self, i, v):
        pass

def output_version_info():
    pass

class SnapshotSystemData_float(object):
    pass

class SnapshotSystemData_double(object):
    pass

class WalltimeLimitReached(object):
    pass

__version__ = "bogus"

def is_MPI_available():
    pass
