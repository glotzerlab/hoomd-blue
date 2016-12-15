# This file exists to allow the hoomd module to import from the source checkout dir
# for use when building the sphinx documentation.

class IntegratorAnisotropicMode(object):
    Automatic = None;
    Anisotropic = None;
    Isotropic = None;

class MuellerPlatheFlow(object):
    class Direction(object):
        X = None;
        Y = None;
        Z = None;
