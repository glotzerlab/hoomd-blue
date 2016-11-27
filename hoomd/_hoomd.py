# This file exists to allow the hoomd module to import from the source checkout dir
# for use when building the sphinx documentation.

class Messenger(object):
    def openPython(self):
        pass

    def notice(self, i, v):
        pass

class GetarCompression(object):
    FastCompress = 1

class GetarDumpMode(object):
    Append = 1
    Overwrite = 1
    OneShot = 1

class GetarProperty(object):
    AngleNames = 1
    AngleTags = 1
    AngleTypes = 1
    AngularMomentum = 1
    Body = 1
    BondNames = 1
    BondTags = 1
    BondTypes = 1
    Box = 1
    Charge = 1
    Diameter = 1
    DihedralNames = 1
    DihedralTags = 1
    DihedralTypes = 1
    Dimensions = 1
    Image = 1
    ImproperNames = 1
    ImproperTags = 1
    ImproperTypes = 1
    Mass = 1
    MomentInertia = 1
    Orientation = 1
    Position = 1
    PotentialEnergy = 1
    Type = 1
    TypeNames = 1
    Velocity = 1
    Virial = 1

class GetarResolution(object):
    Text = 1
    Individual = 1
    Uniform = 1

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
